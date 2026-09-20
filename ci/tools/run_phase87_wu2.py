#!/usr/bin/env python3
"""Run the Phase 87 WU2 FP8 projection probe matrix on one exact GPU.

The HIP probe owns the numerical oracle, candidate launches, and AB-BA-AB
timing protocol.  This runner owns only the fixed matrix, exact-GPU lease,
input identity checks, and fail-closed JSONL validation.  It deliberately
does not impose a wall-clock timeout on a progressing GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import run_mtp_teacher_forced_r9700 as lease


REPO = Path(__file__).resolve().parents[2]
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": lease.UUID}
QWEN_STATUS = "/home/homelab1/.local/bin/qwen38-subagent-server"

# Probe inputs; the build manifest additionally hashes the linked lowp archive.
SOURCE_PATHS = (
    Path("native/hip/tests/phase87_wu2_probe.hip.cpp"),
    Path("native/hip/src/matmul_kernel_internal.hpp"),
    Path("native/hip/src/matmul_kernel.hip.cpp"),
    Path("native/lowp/include/lowp/detail/lowp_kernel_internal.hpp"),
    Path("native/lowp/src/lowp_kernel.hip.cpp"),
    Path("native/hip/tests/phase87_wu2_gemv.hpp"),
    Path("native/hip/tests/phase87_wu2_fused.hpp"),
    Path("native/hip/tests/phase87_wu2_lt_control.hpp"),
    Path("ci/tools/run_phase87_wu2.py"),
    Path("native/lowp/include/lowp/detail/low_precision_block_codec.hpp"),
)

PRODUCTION_SHAPES = (
    (5120, 10240),
    (6144, 5120),
    (5120, 17408),
    (5120, 6144),
    (5120, 12288),
    (17408, 5120),
    (5120, 1024),
    (5120, 248320),
)
BOUNDARY_SHAPES = (
    (31, 7),
    (32, 8),
    (33, 9),
    (127, 31),
    (128, 32),
    (129, 33),
    (511, 255),
    (512, 256),
    (513, 257),
)
CANCELLATION_SHAPES = ((127, 33), (128, 33), (513, 257))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_device(uuid: str) -> Path:
    matches = []
    for unique_id in Path("/sys/class/drm").glob("card[0-9]*/device/unique_id"):
        if unique_id.read_text().strip().lower() == uuid.removeprefix("GPU-").lower():
            matches.append(unique_id.parent)
    if len(matches) != 1:
        raise RuntimeError(f"exact GPU UUID is not unique: {uuid} ({len(matches)})")
    return matches[0]


def source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not (REPO / path).is_file()]
    if missing:
        raise RuntimeError("missing WU2 source: " + ", ".join(map(str, missing)))
    return {str(path): digest(REPO / path) for path in SOURCE_PATHS}


def matrix() -> list[dict[str, int]]:
    cases: list[dict[str, int]] = []
    for k, n in PRODUCTION_SHAPES:
        for m in (1, 2, 3):
            cases.append({"m": m, "k": k, "n": n, "pattern": 0, "bench": 1})
    for k, n in BOUNDARY_SHAPES:
        for m in (1, 2, 3):
            cases.append({"m": m, "k": k, "n": n, "pattern": 0, "bench": 0})
    for k, n in CANCELLATION_SHAPES:
        for m in (1, 3):
            cases.append({"m": m, "k": k, "n": n, "pattern": 1, "bench": 0})
    cases.extend([{"m": 1, "k": 128, "n": 32, "pattern": 2, "bench": 0}, {"m": 3, "k": 129, "n": 33, "pattern": 2, "bench": 0}])
    return cases


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def validate_rows(path: Path, case: dict[str, int], target: str) -> dict:
    rows = read_jsonl(path)
    identity_rows = [row for row in rows if row.get("kind") == "identity"]
    oracle_rows = [row for row in rows if row.get("kind") == "oracle"]
    performance_rows = [row for row in rows if row.get("kind") == "performance"]
    cleanup_rows = [row for row in rows if row.get("kind") == "cleanup"]
    if len(identity_rows) != 1 or len(cleanup_rows) != 1:
        raise RuntimeError(f"identity/cleanup row is missing for {path}")
    identity = identity_rows[0]
    for key in ("target", "m", "k", "n", "pattern"):
        wanted = target if key == "target" else case[key]
        if identity.get(key) != wanted:
            raise RuntimeError(f"identity mismatch for {path}: {identity}")
    if not isinstance(identity.get("control_available"), bool):
        raise RuntimeError(f"control_available is missing for {path}")
    if not isinstance(identity.get("weight_pool_bytes"), int) or identity["weight_pool_bytes"] <= 0:
        raise RuntimeError(f"invalid weight_pool_bytes for {path}")
    if not isinstance(identity.get("copies"), int) or identity["copies"] <= 0:
        raise RuntimeError(f"invalid copies for {path}")

    expected_oracles = 4 if identity["control_available"] else 3
    expected_variants = list(range(4)) if identity["control_available"] else [1, 2, 3]
    if len(oracle_rows) != expected_oracles or sorted(row.get("variant") for row in oracle_rows) != expected_variants:
        raise RuntimeError(f"oracle matrix is incomplete for {path}")
    expected_performance = 3 if case["bench"] else 0
    if len(performance_rows) != expected_performance:
        raise RuntimeError(f"performance matrix is incomplete for {path}")
    if expected_performance and sorted(row.get("variant") for row in performance_rows) != [1, 2, 3]:
        raise RuntimeError(f"performance variants are duplicated or missing for {path}")
    expected_rows = 1 + expected_oracles + expected_performance + 1
    if len(rows) != expected_rows:
        raise RuntimeError(f"unexpected row count for {path}: {len(rows)} != {expected_rows}")

    for row in oracle_rows:
        if row.get("state") != "PASS":
            raise RuntimeError(f"oracle did not PASS for {path}: {row}")
        for key in ("finite", "repeat", "guard", "quantizer_bitwise"):
            if row.get(key) is not True:
                raise RuntimeError(f"oracle {key} failed for {path}: {row}")
        if not isinstance(row.get("max_ulp"), int) or row["max_ulp"] < 0 or row["max_ulp"] > 4:
            raise RuntimeError(f"oracle max_ulp exceeds bound for {path}: {row}")
        if not finite_nonnegative(row.get("max_abs")):
            raise RuntimeError(f"oracle max_abs is invalid for {path}: {row}")

    for row in performance_rows:
        if row.get("variant") not in (1, 2, 3):
            raise RuntimeError(f"unexpected performance variant for {path}: {row}")
        if row.get("warmup_ms") != 300 or row.get("samples") != 27 or row.get("order") != "AB-BA-AB":
            raise RuntimeError(f"performance protocol mismatch for {path}: {row}")
        scalar_keys = (
            "control_quant_ms", "control_dot_ms", "control_total_ms",
            "candidate_quant_ms", "candidate_dot_ms", "candidate_total_ms",
        )
        for key in scalar_keys:
            value = row.get(key)
            if row["variant"] == 2 and key == "candidate_quant_ms":
                if not finite_nonnegative(value):
                    raise RuntimeError(f"C2 candidate quant timing is invalid for {path}: {row}")
            elif not positive(value):
                raise RuntimeError(f"performance timing is invalid for {path}: {row}")
        sample_keys = (
            "control_quant_samples_ms", "control_dot_samples_ms", "control_total_samples_ms",
            "candidate_quant_samples_ms", "candidate_dot_samples_ms", "candidate_total_samples_ms",
        )
        for key in sample_keys:
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 27:
                raise RuntimeError(f"performance samples are missing for {path}: {row}")
            for value in values:
                if row["variant"] == 2 and key == "candidate_quant_samples_ms":
                    valid = finite_nonnegative(value)
                else:
                    valid = positive(value)
                if not valid:
                    raise RuntimeError(f"performance sample is invalid for {path}: {row}")

    cleanup = cleanup_rows[0]
    if cleanup.get("state") != "PASS" or cleanup.get("live_allocations") != 0:
        raise RuntimeError(f"cleanup failed for {path}: {cleanup}")
    return {
        "row_count": len(rows),
        "oracle_count": len(oracle_rows),
        "performance_count": len(performance_rows),
        "control_available": identity["control_available"],
        "cleanup": cleanup,
    }


def save(report: dict, output: Path) -> None:
    temporary = output / "execution.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output / "execution.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=UUIDS, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-m1", action="store_true", help="validate the integrated gfx1201 C3 on its eight admitted shapes")
    args = parser.parse_args()

    binary = args.binary.resolve()
    output = args.output.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"executable binary is required: {binary}")
    output.mkdir(parents=True, exist_ok=False)
    uuid = UUIDS[args.target]
    device = exact_device(uuid)
    level = device / "power_dpm_force_performance_level"
    source_sha256 = source_hashes()
    cases = matrix()
    if args.production_m1:
        if args.target != "gfx1201":
            parser.error("production C3 is exact gfx1201 only")
        cases = [case for case in cases if case["m"] == 1 and case["bench"] == 1]
    report = {
        "schema": "phase87-wu2-execution-v1",
        "state": "running",
        "production_m1": args.production_m1,
        "target": args.target,
        "uuid": uuid,
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "source_sha256": source_sha256,
        "binary_sha256_before": digest(binary),
        "source_sha256_before": source_sha256,
        "identity_check_scope": "source and binary bytes unchanged during this run; this does not prove build identity",
        "matrix": cases,
        "started": time.time(),
        "performance_level_before": level.read_text().strip(),
        "jobs": [],
    }
    save(report, output)
    stopped = False
    try:
        if args.target == "gfx1030":
            status = subprocess.check_output([QWEN_STATUS, "status"], text=True)
            if "process:  stopped" not in status:
                raise RuntimeError("local Qwen must be idle and stopped before GPU measurement")
        else:
            report["service_was_active"] = lease.active()
            report["service_hashes_before"] = lease.service_hashes()
            if report["service_was_active"]:
                clients = subprocess.check_output(
                    ["ss", "-Htn", "state", "established", "( sport = :8000 )"],
                    text=True,
                )
                if clients.strip() or lease.health("/readyz") != 200:
                    raise RuntimeError("resident service busy or unready; cannot lease")
                stopped = True
                report["service_stopped"] = True
                save(report, output)
                subprocess.run(["systemctl", "--user", "stop", lease.UNIT], check=True)
                for _ in range(60):
                    if not lease.active():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("resident service did not stop")

        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("SLLM_") and key not in
            ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        }
        env.update({"ROCR_VISIBLE_DEVICES": uuid, "LD_LIBRARY_PATH": "/opt/rocm/lib"})
        for index, case in enumerate(cases):
            stem = (
                f"case-{index:02d}-m{case['m']}-k{case['k']}-n{case['n']}-"
                f"pattern{case['pattern']}-bench{case['bench']}"
            )
            raw_path = output / f"{stem}.jsonl"
            stderr_path = output / f"{stem}.stderr"
            command = [
                str(binary), "--target", args.target,
                "--m", str(case["m"]), "--k", str(case["k"]),
                "--n", str(case["n"]), "--pattern", str(case["pattern"]),
                "--bench", str(case["bench"]),
            ]
            job = {
                "name": stem,
                "case": case,
                "command": command,
                "raw_jsonl": str(raw_path),
                "stderr": str(stderr_path),
                "started": time.time(),
            }
            report["jobs"].append(job)
            save(report, output)
            with raw_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr)
                job["pid"] = process.pid
                save(report, output)
                print(f"{args.target} {stem} pid={process.pid}", flush=True)
                job["exit_code"] = process.wait()
            job["ended"] = time.time()
            job["raw_sha256"] = digest(raw_path)
            if job["exit_code"] != 0:
                raise RuntimeError(f"probe exited {job['exit_code']}: {stem}")
            job["validation"] = validate_rows(raw_path, case, args.target)
            if args.production_m1 and read_jsonl(raw_path)[0].get("production_dot4") is not True:
                raise RuntimeError("production mode requires the linked production C3 launcher")
            save(report, output)
            print(f"{args.target} {stem}: oracle/cleanup PASS", flush=True)
        report["state"] = "complete"
    except Exception as error:  # noqa: BLE001 - cleanup/evidence is recorded below
        report["state"] = "failed"
        report["error"] = str(error)
    finally:
        if stopped:
            subprocess.run(["systemctl", "--user", "start", lease.UNIT], check=False)
            for _ in range(120):
                if lease.active() and lease.health("/healthz") == lease.health("/readyz") == 200:
                    break
                time.sleep(1)
            report["restored_health"] = lease.health("/healthz")
            report["restored_ready"] = lease.health("/readyz")
        if args.target == "gfx1201" and "service_was_active" in report:
            report["service_hashes_after"] = lease.service_hashes()
            report["service_restored"] = (
                lease.active() == report["service_was_active"]
                and report["service_hashes_before"] == report["service_hashes_after"]
                and (not stopped or report.get("restored_health") == report.get("restored_ready") == 200)
            )
            if not report["service_restored"]:
                report["state"] = "failed"
        try:
            report["binary_sha256_after"] = digest(binary)
            report["source_sha256_after"] = source_hashes()
            report["run_inputs_unchanged"] = (
                report["binary_sha256_before"] == report["binary_sha256_after"]
                and report["source_sha256_before"] == report["source_sha256_after"]
            )
        except Exception as error:  # noqa: BLE001 - preserve mutation-check failure
            report["run_inputs_unchanged"] = False
            report["run_inputs_check_error"] = str(error)
        if not report["run_inputs_unchanged"]:
            report["state"] = "failed"
        report["performance_level_after"] = level.read_text().strip()
        report["performance_level_restored"] = (
            report["performance_level_after"] == report["performance_level_before"]
        )
        if not report["performance_level_restored"]:
            report["state"] = "failed"
        report["ended"] = time.time()
        save(report, output)
    print(report["state"], report.get("error", ""), flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
