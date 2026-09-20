#!/usr/bin/env python3
"""Run the Phase 87 WU1.1 attention operator matrix on one exact GPU.

The probe itself owns the numerical oracle and AB-BA timing protocol.  This
runner owns only the operator matrix, the GPU lease, and fail-closed parsing of
the JSONL emitted by each probe process.  It deliberately does not impose a
wall-clock timeout: a progressing GPU process must be observed to completion.
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
SOURCE_PATHS = (
    Path("native/hip/tests/phase87_wu1_1_probe.hip.cpp"),
    Path("native/hip/tests/phase87_wu1_1_candidates.hpp"),
    Path("native/hip/src/causal_attention_kernel.hip.cpp"),
    Path("native/hip/src/causal_attention_kernel_internal.hpp"),
    Path("native/hip/src/causal_attention_runtime.inc"),
)



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
        raise RuntimeError("missing WU1 source: " + ", ".join(map(str, missing)))
    return {str(path): digest(REPO / path) for path in SOURCE_PATHS}


def matrix() -> list[dict[str, int]]:
    cases = []
    for length in (1023, 1024, 1025, 8192, 8193, 8256):
        for query_count in (1, 2, 3):
            cases.append({"length": length, "m": query_count, "pattern": 0, "bench": 1})
    # Match the fixed WU0 read reference / decode mean context for the cutoff.
    # Pattern 1 exercises the causal tail at both requested boundaries.  The
    # stronger pattern 2 scales Q/K to expose softmax range handling as well.
    for pattern in (1,):
        for length in (1025, 8193):
            for query_count in (1, 2, 3):
                cases.append({"length": length, "m": query_count, "pattern": pattern, "bench": 0})
    for length in (1025, 8193):
        cases.append({"length": length, "m": 3, "pattern": 2, "bench": 0})
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


def validate_rows(path: Path, case: dict[str, int], target: str) -> dict:
    rows = read_jsonl(path)
    oracle_rows = [row for row in rows if row.get("kind") == "oracle"]
    performance_rows = [row for row in rows if row.get("kind") == "performance"]
    cleanup_rows = [row for row in rows if row.get("kind") == "cleanup"]
    variant_count = 3 if target == "gfx1030" else 4
    expected_performance = variant_count - 1 if case["bench"] else 0
    if len(rows) != variant_count + expected_performance + 1:
        raise RuntimeError(f"unexpected row count for {path}: {len(rows)}")
    if len(oracle_rows) != variant_count or sorted(row.get("variant") for row in oracle_rows) != list(range(variant_count)):
        raise RuntimeError(f"oracle matrix is incomplete for {path}")
    if len(performance_rows) != expected_performance:
        raise RuntimeError(f"performance matrix is incomplete for {path}")
    if expected_performance and sorted(row.get("variant") for row in performance_rows) != list(range(1, variant_count)):
        raise RuntimeError(f"performance variants are duplicated or missing for {path}")
    if len(cleanup_rows) != 1:
        raise RuntimeError(f"cleanup row is missing for {path}")

    for row in oracle_rows:
        for key in ("target", "length", "m", "pattern"):
            if row.get(key) != (target if key == "target" else case[key]):
                raise RuntimeError(f"oracle identity mismatch for {path}: {row}")
        if row.get("state") != "PASS" or row.get("repeat") is not True:
            raise RuntimeError(f"oracle did not PASS repeat check for {path}: {row}")
        if not isinstance(row.get("control_bitwise"), bool):
            raise RuntimeError(f"oracle control comparison is missing for {path}: {row}")
        if row.get("guard") is not True or row.get("finite") is not True:
            raise RuntimeError(f"oracle guard/finite check failed for {path}: {row}")
        for key in ("max_ulp", "max_abs", "max_relative"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise RuntimeError(f"invalid oracle metric for {path}: {row}")
        if (row["max_ulp"] > 4 or row["max_abs"] > 0.03125 or
                row["max_relative"] > 0.04):
            raise RuntimeError(f"oracle metric exceeds WU1 acceptance bound for {path}: {row}")

    for row in performance_rows:
        for key in ("target", "length", "m", "pattern"):
            if row.get(key) != (target if key == "target" else case[key]):
                raise RuntimeError(f"performance identity mismatch for {path}: {row}")
        if row.get("variant") not in range(1, variant_count):
            raise RuntimeError(f"unexpected performance variant for {path}: {row}")
        if (row.get("samples_per_variant") != 27 or row.get("warmup_ms") != 300 or
                row.get("order") != "AB-BA-AB"):
            raise RuntimeError(f"performance protocol mismatch for {path}: {row}")
        for key in ("control_stage1_ms", "candidate_stage1_ms", "control_stage2_ms", "candidate_stage2_ms", "control_total_ms", "candidate_total_ms"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise RuntimeError(f"invalid performance metric for {path}: {row}")
        for key in ("control_stage1_samples_ms", "candidate_stage1_samples_ms",
                    "control_stage2_samples_ms", "candidate_stage2_samples_ms",
                    "control_total_samples_ms", "candidate_total_samples_ms"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 27 or any(
                    not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0
                    for value in values):
                raise RuntimeError(f"invalid performance samples for {path}: {row}")

    cleanup = cleanup_rows[0]
    if cleanup.get("state") != "PASS" or cleanup.get("live_allocations") != 0:
        raise RuntimeError(f"cleanup failed for {path}: {cleanup}")
    return {"row_count": len(rows), "oracle_count": len(oracle_rows),
            "performance_count": len(performance_rows), "cleanup": cleanup}


def save(report: dict, output: Path) -> None:
    temporary = output / "execution.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output / "execution.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=UUIDS, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    report = {
        "schema": "phase87-wu1-1-execution-v1",
        "state": "running",
        "target": args.target,
        "uuid": uuid,
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "source_sha256": source_sha256,
        "binary_sha256_before": digest(binary),
        "source_sha256_before": source_sha256,
        "identity_check_scope":
            "source and binary bytes unchanged during this run; this does not prove build identity",
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
                    ["ss", "-Htn", "state", "established", "( sport = :8000 )"], text=True)
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

        env = {key: value for key, value in os.environ.items()
               if not key.startswith("SLLM_") and key not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        env.update({"ROCR_VISIBLE_DEVICES": uuid, "LD_LIBRARY_PATH": "/opt/rocm/lib"})
        for index, case in enumerate(cases):
            stem = (f"case-{index:02d}-length{case['length']}-m{case['m']}-"
                    f"pattern{case['pattern']}-bench{case['bench']}")
            raw_path = output / f"{stem}.jsonl"
            stderr_path = output / f"{stem}.stderr"
            command = [str(binary), "--length", str(case["length"]), "--m", str(case["m"]),
                       "--pattern", str(case["pattern"]), "--bench", str(case["bench"]),
                       "--target", args.target]
            job = {"name": stem, "case": case, "command": command,
                   "raw_jsonl": str(raw_path), "stderr": str(stderr_path),
                   "started": time.time()}
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
            save(report, output)
            print(f"{args.target} {stem}: oracle/cleanup PASS", flush=True)
        report["state"] = "complete"
    except Exception as error:  # noqa: BLE001 - cleanup and evidence are recorded below
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
                and (not stopped or report.get("restored_health") == report.get("restored_ready") == 200))
            if not report["service_restored"]:
                report["state"] = "failed"
        try:
            report["binary_sha256_after"] = digest(binary)
            report["source_sha256_after"] = source_hashes()
            report["run_inputs_unchanged"] = (
                report["binary_sha256_before"] == report["binary_sha256_after"] and
                report["source_sha256_before"] == report["source_sha256_after"])
        except Exception as error:  # noqa: BLE001 - preserve mutation-check failure in evidence
            report["run_inputs_unchanged"] = False
            report["run_inputs_check_error"] = str(error)
        if not report["run_inputs_unchanged"]:
            report["state"] = "failed"
        report["performance_level_after"] = level.read_text().strip()
        report["performance_level_restored"] = (
            report["performance_level_after"] == report["performance_level_before"])
        if not report["performance_level_restored"]:
            report["state"] = "failed"
        report["ended"] = time.time()
        save(report, output)
    print(report["state"], report.get("error", ""), flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
