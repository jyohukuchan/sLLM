#!/usr/bin/env python3
"""Measure CPU-only SQ8 artifact-F32 scaling without capture payload I/O.

Each configuration starts one strict-F32 reference process per explicitly
provided CPU set.  The reference binary is run with ``--no-capture`` so the
measurement writes only small receipts/logs and cannot overwrite a resumable
corpus checkpoint.  GPU runtime visibility is disabled for every child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCHEMA_VERSION = "ullm.sq8.artifact_fp32_reference.smt_scaling_measurement.v1"
GPU_INVISIBLE_ENVIRONMENT = {
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
    "ULLM_HIP_VISIBLE_DEVICES": "-1",
    "CUDA_VISIBLE_DEVICES": "",
}


def parse_cpu_set(value: str) -> set[int]:
    members: set[int] = set()
    for part in value.split(","):
        if not part:
            raise ValueError(f"empty CPU-set component in {value!r}")
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first > last:
                raise ValueError(f"descending CPU range {part!r}")
            members.update(range(first, last + 1))
        else:
            members.add(int(part))
    return members


def cpu_topology(cpu_ids: set[int]) -> dict[str, str]:
    topology: dict[str, str] = {}
    for cpu in sorted(cpu_ids):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            topology[str(cpu)] = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"cannot read SMT topology for CPU {cpu}: {error}") from error
    return topology


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def atomic_json(path: Path, value: Any) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def max_rss_kib(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpu-sets", required=True)
    parser.add_argument("--forwards", type=int, default=20)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/ullm-sq8-fp32-reference"),
    )
    parser.add_argument("--minimum-forward-seconds", type=float, default=180.0)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> list[str]:
    if args.threads < 1 or args.threads > 128:
        raise ValueError("--threads must be in 1..=128")
    if args.forwards < 1 or args.forwards > 4096:
        raise ValueError("--forwards must be in 1..=4096")
    if not -20 <= args.nice <= 19:
        raise ValueError("--nice must be in -20..=19")
    if args.minimum_forward_seconds < 0:
        raise ValueError("--minimum-forward-seconds must be non-negative")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    for name in ("artifact", "package", "binary"):
        path = getattr(args, name)
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    values = args.cpu_sets.split(";")
    if not values or any(not value for value in values):
        raise ValueError("--cpu-sets must contain one nonempty set per process")
    available = set(os.sched_getaffinity(0))
    used: set[int] = set()
    for value in values:
        members = parse_cpu_set(value)
        if len(members) != args.threads:
            raise ValueError(
                f"CPU set {value!r} has {len(members)} CPUs, expected {args.threads}"
            )
        unavailable = sorted(members.difference(available))
        if unavailable:
            raise ValueError(f"CPU set {value!r} includes unavailable CPUs {unavailable}")
        overlap = sorted(used.intersection(members))
        if overlap:
            raise ValueError(f"CPU set {value!r} overlaps earlier sets at {overlap}")
        used.update(members)
    return values


def run(args: argparse.Namespace, cpu_sets: list[str]) -> dict[str, Any]:
    output = args.output
    memory_before = mem_available_kib()
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    environment = os.environ.copy()
    environment.update(GPU_INVISIBLE_ENVIRONMENT)
    process_records: list[dict[str, Any]] = []
    launched: list[tuple[subprocess.Popen[bytes], dict[str, Any]]] = []
    started_monotonic = time.monotonic()
    started_unix_seconds = time.time()
    for index, cpu_set in enumerate(cpu_sets):
        worker_output = output / f"proc-{index:02d}"
        command = [
            "taskset",
            "-c",
            cpu_set,
            "nice",
            "-n",
            str(args.nice),
            str(args.binary),
            str(args.artifact),
            str(args.package),
            str(worker_output),
            "--token-id",
            "1",
            "--forwards",
            str(args.forwards),
            "--threads",
            str(args.threads),
            "--max-context",
            "4096",
            "--no-capture",
        ]
        stdout_path = logs / f"proc-{index:02d}.stdout.log"
        stderr_path = logs / f"proc-{index:02d}.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        record: dict[str, Any] = {
            "index": index,
            "cpu_set": cpu_set,
            "command": command,
            "pid": process.pid,
            "started_unix_seconds": time.time(),
            "max_observed_rss_kib": 0,
        }
        process_records.append(record)
        launched.append((process, record))

    while launched:
        still_running: list[tuple[subprocess.Popen[bytes], dict[str, Any]]] = []
        for process, record in launched:
            rss = max_rss_kib(process.pid)
            if rss is not None:
                record["max_observed_rss_kib"] = max(record["max_observed_rss_kib"], rss)
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, record))
            else:
                record["return_code"] = return_code
                record["finished_unix_seconds"] = time.time()
        launched = still_running
        if launched:
            time.sleep(0.5)

    finished_monotonic = time.monotonic()
    worker_receipts: list[dict[str, Any]] = []
    for record in process_records:
        if record.get("return_code") != 0:
            raise RuntimeError(f"worker {record['index']} exited {record.get('return_code')}")
        receipt_path = output / f"proc-{record['index']:02d}" / "run.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        steps = receipt.get("forward_steps")
        if not isinstance(steps, list) or len(steps) != args.forwards:
            raise RuntimeError(f"worker {record['index']} receipt has unexpected forward count")
        forward_seconds = sum(float(step["elapsed_seconds"]) for step in steps)
        summaries = [step["summary"] for step in steps]
        worker_receipts.append(
            {
                "index": record["index"],
                "receipt_path": str(receipt_path),
                "initialization_elapsed_seconds": receipt["initialization_elapsed_seconds"],
                "peak_rss_kib": receipt["peak_rss_kib"],
                "forward_elapsed_seconds": forward_seconds,
                "forward_summary_sequence_sha256": sha256_json(summaries),
                "first_forward_summary": summaries[0],
                "last_forward_summary": summaries[-1],
            }
        )
    summary_hashes = {worker["forward_summary_sequence_sha256"] for worker in worker_receipts}
    forward_seconds = [worker["forward_elapsed_seconds"] for worker in worker_receipts]
    total_forwards = args.forwards * len(cpu_sets)
    critical_forward_seconds = max(forward_seconds)
    min_forward_seconds = min(forward_seconds)
    status = "complete" if min_forward_seconds >= args.minimum_forward_seconds else "too_short"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "name": args.name,
        "execution_backend": "cpu_only_no_runtime_context",
        "gpu_environment": GPU_INVISIBLE_ENVIRONMENT,
        "reference_binary": {
            "path": str(args.binary),
            "sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        },
        "artifact": str(args.artifact),
        "package": str(args.package),
        "threads_per_process": args.threads,
        "processes": len(cpu_sets),
        "cpu_sets": cpu_sets,
        "used_logical_cpu_count": sum(len(parse_cpu_set(value)) for value in cpu_sets),
        "cpu_thread_siblings": cpu_topology(set().union(*(parse_cpu_set(value) for value in cpu_sets))),
        "forwards_per_process": args.forwards,
        "total_forwards": total_forwards,
        "nice": args.nice,
        "started_unix_seconds": started_unix_seconds,
        "wall_elapsed_seconds": finished_monotonic - started_monotonic,
        "wall_forwards_per_second": total_forwards / (finished_monotonic - started_monotonic),
        "critical_path_forward_seconds": critical_forward_seconds,
        "minimum_process_forward_seconds": min_forward_seconds,
        "steady_state_forwards_per_second": total_forwards / critical_forward_seconds,
        "minimum_forward_seconds_required": args.minimum_forward_seconds,
        "semantic_result_sequences_identical_across_processes": len(summary_hashes) == 1,
        "distinct_forward_summary_sequence_count": len(summary_hashes),
        "worker_receipts": worker_receipts,
        "processes_detail": process_records,
        "mem_available_kib_before": memory_before,
    }


def main() -> int:
    args = parse_args()
    try:
        cpu_sets = validate(args)
        result = run(args, cpu_sets)
        result["mem_available_kib_after"] = mem_available_kib()
        atomic_json(args.output / "measurement.json", result)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("name", "status", "steady_state_forwards_per_second", "wall_forwards_per_second")}, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
