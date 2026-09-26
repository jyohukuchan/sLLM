#!/usr/bin/env python3
"""Build and run the Phase 87 Stage 1 ID84 SGPR scalarization probe."""

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

import run_mtp_teacher_forced_r9700 as r9700_lease


REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("native/hip/tests/phase87_stage1_nvfp4_sgpr_probe.hip.cpp")
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": "GPU-a8e9ddefa2d60f55"}
QWEN_STATUS = "/home/homelab1/.local/bin/qwen38-subagent-server"
ROCM_COMPILER = "/opt/rocm/bin/amdclang++"
SOURCE_PATHS = (
    SOURCE,
    Path("native/lowp/src/nvfp4_decode_scale_lut.inc"),
    Path("native/hip/src/matmul_kernel_internal.hpp"),
    Path("native/lowp/include/lowp/detail/lowp_kernel_internal.hpp"),
    Path("ci/tools/run_phase87_stage1_sgpr.py"),
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


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


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


def validate_case(path: Path, target: str, shape: tuple[int, int], benchmark: bool) -> dict:
    rows = read_jsonl(path)
    identities = [row for row in rows if row.get("kind") == "identity"]
    oracles = [row for row in rows if row.get("kind") == "oracle"]
    performances = [row for row in rows if row.get("kind") == "performance"]
    cleanups = [row for row in rows if row.get("kind") == "cleanup"]
    if len(identities) != 1 or len(oracles) != 1 or len(cleanups) != 1:
        raise RuntimeError(f"identity/oracle/cleanup rows incomplete: {path}")
    identity = identities[0]
    wanted = {"target": target, "m": 1, "k": shape[0], "n": shape[1]}
    if any(identity.get(key) != value for key, value in wanted.items()):
        raise RuntimeError(f"identity mismatch for {path}: {identity}")
    if (
        identity.get("candidate") != "id84_sgpr_scalarized"
        or identity.get("control_launcher") != "production_id84"
        or identity.get("control_variant") != 84
        or identity.get("gpu_execution") is not True
        or identity.get("fallback_used") is not False
        or not isinstance(identity.get("weight_pool_bytes"), int)
        or identity["weight_pool_bytes"] < 512 * 2**20
    ):
        raise RuntimeError(f"identity contract failed for {path}: {identity}")
    oracle = oracles[0]
    if oracle.get("state") != "PASS" or any(
        oracle.get(key) is not True
        for key in ("finite", "guard", "repeat", "candidate_control_bitwise")
    ):
        raise RuntimeError(f"oracle contract failed for {path}: {oracle}")
    if not isinstance(oracle.get("max_ulp"), int) or oracle["max_ulp"] > 4:
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
            if not finite_positive(performance.get(key)):
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
                not finite_positive(value) for value in samples
            ):
                raise RuntimeError(f"invalid raw timing samples for {path}: {key}")
    if cleanups[0].get("state") != "PASS" or cleanups[0].get("live_allocations") != 0:
        raise RuntimeError(f"cleanup failed for {path}: {cleanups[0]}")
    if len(rows) != 3 + expected_performance:
        raise RuntimeError(f"unexpected row count for {path}: {len(rows)}")
    return {"identity": identity, "oracle": oracle, "performance": performances}


def build_command(target: str, archive: Path, binary: Path) -> list[str]:
    return [
        ROCM_COMPILER,
        "-D__HIP_ROCclr__=1",
        f'-DSLLM_HIP_COMPILE_TARGET="{target}"',
        "-O3",
        "-ffp-contract=off",
        "-std=gnu++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"--offload-arch={target}",
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
    parser.add_argument("--target", choices=tuple(UUIDS), required=True)
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
    device_path = exact_device(UUIDS[args.target])
    output.mkdir(parents=True)
    shape_cases = ((0, (5120, 17408)), (1, (17408, 5120)))
    if args.shape == "wide":
        shape_cases = (shape_cases[0],)
    elif args.shape == "down":
        shape_cases = (shape_cases[1],)
    binary = output / f"phase87-stage1-sgpr-{args.target}"
    before = source_hashes(archive)
    command = build_command(args.target, archive, binary)
    build = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    (output / "build.stdout").write_text(build.stdout)
    (output / "build.stderr").write_text(build.stderr)
    if build.returncode != 0:
        print(f"build failed; see {output / 'build.stderr'}", file=sys.stderr)
        return 2
    service_status = subprocess.check_output([QWEN_STATUS, "status"], text=True)
    if args.target == "gfx1030" and "process:  stopped" not in service_status:
        raise RuntimeError("local Qwen must be stopped before gfx1030 measurement")
    service_was_active = False
    service_stopped = False
    service_hashes_before = None
    service_hashes_after = None
    service_restored = True
    if args.target == "gfx1201":
        clients = subprocess.check_output(
            ["ss", "-Htn", "state", "established", "( sport = :8000 )"],
            text=True,
        )
        if clients.strip():
            raise RuntimeError("active R9700 service connection; cannot lease")
        service_was_active = r9700_lease.active()
        service_hashes_before = r9700_lease.service_hashes()
        if service_was_active:
            if r9700_lease.health("/readyz") != 200:
                raise RuntimeError("R9700 service is active but not ready")
            subprocess.run(["systemctl", "--user", "stop", r9700_lease.UNIT], check=True)
            service_stopped = True
            for _ in range(60):
                if not r9700_lease.active():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("R9700 service did not stop")
    level_path = device_path / "power_dpm_force_performance_level"
    level_before = level_path.read_text().strip()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLLM_")
        and key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }
    env.update(
        {
            "ROCR_VISIBLE_DEVICES": UUIDS[args.target],
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "0"
            if args.target == "gfx1030"
            else env.get("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING", "1"),
        }
    )
    jobs: list[dict] = []
    state = "complete"
    error = None
    started = time.time()
    try:
        for shape_index, shape in shape_cases:
            label = f"shape{shape_index}-k{shape[0]}-n{shape[1]}"
            raw_path = output / f"{label}.jsonl"
            stderr_path = output / f"{label}.stderr"
            command_run = [
                str(binary),
                "--target",
                args.target,
                "--shape",
                str(shape_index),
                "--bench",
                "0" if args.no_bench else "1",
            ]
            job = {
                "shape": {"k": shape[0], "n": shape[1]},
                "command": command_run,
                "raw_jsonl": str(raw_path),
                "stderr": str(stderr_path),
                "started": time.time(),
            }
            jobs.append(job)
            with raw_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command_run, cwd=REPO, env=env, stdout=stdout, stderr=stderr
                )
                job["pid"] = process.pid
                job["exit_code"] = process.wait()
            job["ended"] = time.time()
            job["raw_sha256"] = digest(raw_path)
            if job["exit_code"] != 0:
                raise RuntimeError(f"probe exited {job['exit_code']}: {label}")
            job["validation"] = validate_case(raw_path, args.target, shape, not args.no_bench)
    except Exception as exc:  # noqa: BLE001 - preserve fail-closed report
        state = "failed"
        error = str(exc)
    finally:
        if args.target == "gfx1201" and service_stopped:
            subprocess.run(["systemctl", "--user", "start", r9700_lease.UNIT], check=False)
            for _ in range(120):
                if r9700_lease.active() and r9700_lease.health("/healthz") == 200 and r9700_lease.health("/readyz") == 200:
                    break
                time.sleep(1)
        if args.target == "gfx1201":
            service_hashes_after = r9700_lease.service_hashes()
            service_restored = (
                r9700_lease.active() == service_was_active
                and service_hashes_after == service_hashes_before
                and (not service_stopped or r9700_lease.health("/healthz") == r9700_lease.health("/readyz") == 200)
            )
    level_after = level_path.read_text().strip()
    after = source_hashes(archive)
    report = {
        "schema": "phase87-stage1-sgpr-execution-v1",
        "state": state,
        "target": args.target,
        "uuid": UUIDS[args.target],
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
        "shapes": [{"index": index, "k": shape[0], "n": shape[1]} for index, shape in shape_cases],
        "service_was_active": service_was_active,
        "service_stopped": service_stopped,
        "service_hashes_before": service_hashes_before,
        "service_hashes_after": service_hashes_after,
        "service_restored": service_restored,
        "jobs": jobs,
        "started": started,
        "ended": time.time(),
    }
    if error is not None:
        report["error"] = error
    if not report["performance_level_restored"] or not report["run_inputs_unchanged"] or not report["service_restored"]:
        report["state"] = "failed"
    (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Phase87 Stage1 SGPR {args.target}: {report['state']} jobs={len(jobs)}/{len(shape_cases)}")
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
