#!/usr/bin/env python3
"""Launch the CPU-only resumable SQ8 artifact-FP32 corpus workers.

Each worker owns one causally dependent corpus case.  This launcher only
parallelizes independent cases, keeps every CPU set disjoint, and defaults to
physical Threadripper cores (logical CPUs 0--63).  An explicit opt-in permits
SMT siblings (logical CPUs 64--127).  It does not start, stop, or query a
service and it never invokes a GPU binary.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


FROZEN_GATE_SHA256 = "64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf"
LAUNCHER_SCHEMA_VERSION = "ullm.sq8.artifact_fp32_reference.parallel_launcher.v1"
PHYSICAL_CPU_COUNT = 64
LOGICAL_CPU_COUNT = 128


@dataclasses.dataclass(frozen=True)
class Job:
    mode: str
    case_id: str

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.case_id}"


JOBS: tuple[Job, ...] = (
    Job("sequential_m1", "raw-p0001-g1024"),
    Job("sequential_m1", "raw-p0008-g512"),
    Job("sequential_m1", "raw-p0032-g512"),
    Job("sequential_m1", "raw-p0128-g512"),
    Job("sequential_m1", "raw-p0512-g512"),
    Job("sequential_m1", "chat-p2048-g512"),
    Job("sequential_m1", "chat-p3584-g512"),
    Job("sequential_m1", "raw-p0127-g4"),
    Job("sequential_m1", "raw-p0255-g4"),
    Job("sequential_m1", "raw-p0511-g4"),
    Job("sequential_m1", "raw-p1023-g4"),
    Job("sequential_m1", "raw-p4095-g1"),
    Job("m128_chunks_with_declared_tail", "raw-p0128-g512"),
    Job("m128_chunks_with_declared_tail", "raw-p0512-g512"),
    Job("m128_chunks_with_declared_tail", "chat-p2048-g512"),
    Job("m128_chunks_with_declared_tail", "chat-p3584-g512"),
    Job("m128_chunks_with_declared_tail", "raw-p4095-g1"),
)

# These are the frozen corpus's prompt-plus-forced-decode forward counts.  They
# are scheduling data only; corpus construction and every input hash remain
# validated by the Rust worker against the frozen gate JSON.
CASE_FORWARD_COUNTS = {
    "raw-p0001-g1024": 1_025,
    "raw-p0008-g512": 520,
    "raw-p0032-g512": 544,
    "raw-p0128-g512": 640,
    "raw-p0512-g512": 1_024,
    "chat-p2048-g512": 2_560,
    "chat-p3584-g512": 4_096,
    "raw-p0127-g4": 131,
    "raw-p0255-g4": 259,
    "raw-p0511-g4": 515,
    "raw-p1023-g4": 1_027,
    "raw-p4095-g1": 4_096,
}
ORDERED_JOBS = tuple(
    sorted(JOBS, key=lambda job: (-CASE_FORWARD_COUNTS[job.case_id], job.mode, job.case_id))
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def contiguous_cpu_set(slot: int, threads: int) -> str:
    first = slot * threads
    last = first + threads - 1
    return f"{first}-{last}"


def expand_cpu_set(value: str) -> set[int]:
    members: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first > last:
                raise ValueError(f"invalid descending CPU range {part!r}")
            members.update(range(first, last + 1))
        else:
            members.add(int(part))
    return members


def affinity_sets(args: argparse.Namespace) -> list[str]:
    if args.cpu_sets is None:
        return [contiguous_cpu_set(slot, args.threads) for slot in range(args.processes)]
    values = args.cpu_sets.split(";")
    if len(values) != args.processes:
        raise ValueError(
            "--cpu-sets must contain exactly one semicolon-separated CPU set per process"
        )
    used: set[int] = set()
    for value in values:
        members = expand_cpu_set(value)
        if len(members) != args.threads:
            raise ValueError(
                f"CPU set {value!r} has {len(members)} CPUs, expected {args.threads}"
            )
        if any(cpu < 0 or cpu >= LOGICAL_CPU_COUNT for cpu in members):
            raise ValueError(f"CPU set {value!r} is outside logical CPU range 0--127")
        if used.intersection(members):
            raise ValueError(f"CPU set {value!r} overlaps another requested worker set")
        used.update(members)
    return values


def case_output(root: Path, job: Job) -> Path:
    return root / "cases" / job.mode / job.case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/ullm-sq8-fp32-reference-corpus"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json"),
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument(
        "--cpu-sets",
        help="semicolon-separated taskset CPU sets; defaults to contiguous physical-core partitions",
    )
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--rss-budget-kib-per-process", type=int, default=786_432)
    parser.add_argument("--memory-reserve-kib", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-smt",
        action="store_true",
        help="permit explicitly supplied SMT sibling logical CPUs 64--127",
    )
    parser.add_argument(
        "--verify-resume",
        action="store_true",
        help="read-only validation of a --resume invocation; writes no launcher files",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.threads <= LOGICAL_CPU_COUNT:
        raise ValueError(f"--threads must be within 1..={LOGICAL_CPU_COUNT}")
    if not 1 <= args.processes <= LOGICAL_CPU_COUNT:
        raise ValueError(f"--processes must be within 1..={LOGICAL_CPU_COUNT}")
    if args.cpu_sets is None and args.threads * args.processes > PHYSICAL_CPU_COUNT:
        raise ValueError(
            "workers would oversubscribe the default physical-core set 0--63: "
            f"{args.threads} threads × {args.processes} processes"
        )
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    if not -20 <= args.nice <= 19:
        raise ValueError("--nice must be in -20..19")
    requested_cpu_sets = affinity_sets(args)
    uses_smt = any(cpu >= PHYSICAL_CPU_COUNT for value in requested_cpu_sets for cpu in expand_cpu_set(value))
    if uses_smt and not args.allow_smt:
        raise ValueError("SMT logical CPUs 64--127 require explicit --allow-smt")
    if args.verify_resume and not args.resume:
        raise ValueError("--verify-resume requires --resume")
    for name in ("artifact", "package", "binary", "gate"):
        path = getattr(args, name)
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    gate_hash = sha256_file(args.gate)
    if gate_hash != FROZEN_GATE_SHA256:
        raise ValueError(
            f"frozen gate SHA-256 mismatch: expected={FROZEN_GATE_SHA256} actual={gate_hash}"
        )
    available = memory_available_kib()
    requested = args.rss_budget_kib_per_process * args.processes
    required = requested + args.memory_reserve_kib
    if required > available:
        raise ValueError(
            "memory preflight failed: reserve + worker RSS budget is "
            f"{required} KiB, MemAvailable is {available} KiB"
        )


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "execution_backend": "cpu_only_no_runtime_context",
        "gpu_environment": {
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
            "ULLM_HIP_VISIBLE_DEVICES": "-1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        "frozen_gate": {
            "path": str(args.gate),
            "sha256": sha256_file(args.gate),
        },
        "artifact": str(args.artifact),
        "package": str(args.package),
        "worker_binary": {
            "path": str(args.binary),
            "sha256": sha256_file(args.binary),
        },
        "seed": 0,
        "threads_per_process": args.threads,
        "processes": args.processes,
        "physical_cpu_set": "0-63",
        "logical_cpu_set": "0-127",
        "allow_smt": args.allow_smt,
        "affinities": affinity_sets(args),
        "nice": args.nice,
        "memory_preflight": {
            "rss_budget_kib_per_process": args.rss_budget_kib_per_process,
            "workers_rss_budget_kib": args.rss_budget_kib_per_process * args.processes,
            "reserve_kib": args.memory_reserve_kib,
        },
        "jobs": [dataclasses.asdict(job) for job in ORDERED_JOBS],
    }


def resume_identity(plan: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that can change corpus bytes or worker checkpoint validity.

    Process count, CPU placement, niceness, and the conservative RSS preflight
    are scheduling controls.  They are recorded per invocation but deliberately
    do not invalidate a checkpoint.  Thread count remains immutable because a
    case worker binds it into its own exact ``plan.json``.
    """

    fields = (
        "execution_backend",
        "gpu_environment",
        "frozen_gate",
        "artifact",
        "package",
        "worker_binary",
        "seed",
        "threads_per_process",
        "jobs",
    )
    try:
        return {field: plan[field] for field in fields}
    except KeyError as error:
        raise ValueError(f"launcher plan is missing resume identity field {error.args[0]!r}") from error


def validate_existing_case_thread_counts(args: argparse.Namespace) -> int:
    """Fail before launching if a checkpoint binds a different thread count."""

    checked = 0
    for job in ORDERED_JOBS:
        path = case_output(args.output, job) / "plan.json"
        if not path.exists():
            continue
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read existing case plan {path}: {error}") from error
        thread_count = plan.get("thread_count")
        if thread_count != args.threads:
            raise ValueError(
                f"existing case plan {path} binds thread_count={thread_count!r}, "
                f"requested --threads={args.threads}"
            )
        checked += 1
    return checked


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = make_plan(args)
    plan_path = args.output / "launcher-plan.json"
    if plan_path.exists():
        if not args.resume:
            raise ValueError(
                f"output already has a launcher plan: {plan_path}; use --resume after reviewing it"
            )
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if resume_identity(existing) != resume_identity(plan):
            raise ValueError(
                "existing launcher corpus identity does not match this resume invocation"
            )
    else:
        if args.verify_resume:
            raise ValueError(f"cannot verify resume without existing launcher plan: {plan_path}")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(
                f"output exists and is nonempty without launcher plan: {args.output}"
            )
        args.output.mkdir(parents=True, exist_ok=True)
        write_new_json(plan_path, plan)
    return plan


def record_execution_invocation(args: argparse.Namespace, plan: dict[str, Any]) -> Path:
    """Keep an immutable audit record when a compatible resume changes scheduling."""

    directory = args.output / "launcher-invocations"
    sequence = f"invocation-{time.time_ns()}-pid-{os.getpid()}.json"
    path = directory / sequence
    write_new_json(
        path,
        {
            "schema_version": "ullm.sq8.artifact_fp32_reference.parallel_launcher.invocation.v1",
            "created_unix_seconds": int(time.time()),
            "resume": args.resume,
            "resume_identity": resume_identity(plan),
            "execution": {
                "threads_per_process": args.threads,
                "processes": args.processes,
                "logical_cpu_set": "0-127",
                "allow_smt": args.allow_smt,
                "affinities": affinity_sets(args),
                "nice": args.nice,
                "memory_preflight": plan["memory_preflight"],
            },
        },
    )
    return path


def worker_command(args: argparse.Namespace, job: Job, slot: int, resume_case: bool) -> list[str]:
    command = [
        "taskset",
        "-c",
        affinity_sets(args)[slot],
        "nice",
        "-n",
        str(args.nice),
        str(args.binary),
        str(args.artifact),
        str(args.package),
        str(case_output(args.output, job)),
        "--gate",
        str(args.gate),
        "--case",
        job.case_id,
        "--mode",
        job.mode,
        "--threads",
        str(args.threads),
        "--expected-gate-sha256",
        FROZEN_GATE_SHA256,
    ]
    if resume_case:
        command.append("--resume")
    return command


def is_complete(output: Path) -> bool:
    receipt = output / "run.json"
    if not receipt.exists():
        return False
    try:
        return json.loads(receipt.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def progress_snapshot(
    args: argparse.Namespace,
    plan: dict[str, Any],
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    completed = sum(state["status"] == "complete" for state in states.values())
    running = sum(state["status"] == "running" for state in states.values())
    failed = sum(state["status"] == "failed" for state in states.values())
    return {
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "status": "failed"
        if failed
        else ("complete" if completed == len(ORDERED_JOBS) else "running"),
        "updated_unix_seconds": int(time.time()),
        "completed_jobs": completed,
        "running_jobs": running,
        "failed_jobs": failed,
        "total_jobs": len(ORDERED_JOBS),
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ).hexdigest(),
        "jobs": states,
    }


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        plan = prepare_plan(args)
        checked_case_plans = validate_existing_case_thread_counts(args) if args.resume else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.verify_resume:
        print(
            json.dumps(
                {
                    "status": "resume_compatible",
                    "checked_case_plans": checked_case_plans,
                    "resume_identity": resume_identity(plan),
                },
                sort_keys=True,
            )
        )
        return 0
    record_execution_invocation(args, plan)
    available = memory_available_kib()
    atomic_json(
        args.output / "launcher-preflight.json",
        {
            "schema_version": LAUNCHER_SCHEMA_VERSION,
            "checked_unix_seconds": int(time.time()),
            "mem_available_kib": available,
            "workers_rss_budget_kib": args.rss_budget_kib_per_process * args.processes,
            "reserve_kib": args.memory_reserve_kib,
            "remaining_after_budget_kib": available
            - args.rss_budget_kib_per_process * args.processes
            - args.memory_reserve_kib,
        },
    )

    states: dict[str, dict[str, Any]] = {}
    pending: list[Job] = []
    for job in ORDERED_JOBS:
        output = case_output(args.output, job)
        if is_complete(output):
            states[job.key] = {"status": "complete", "output": str(output), "resumed": True}
        else:
            states[job.key] = {"status": "pending", "output": str(output), "resumed": output.exists()}
            pending.append(job)

    if args.dry_run:
        for slot, job in enumerate(pending[: args.processes]):
            print(
                " ".join(
                    worker_command(args, job, slot, case_output(args.output, job).exists())
                )
            )
        atomic_json(args.output / "launcher-progress.json", progress_snapshot(args, plan, states))
        return 0

    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(plan["gpu_environment"])
    running: dict[str, tuple[subprocess.Popen[bytes], Job, int]] = {}
    stopping = False

    def request_stop(_signal_number: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while pending or running:
        while not stopping and pending and len(running) < args.processes:
            job = pending.pop(0)
            used_slots = {slot for _, _, slot in running.values()}
            slot = next(index for index in range(args.processes) if index not in used_slots)
            output = case_output(args.output, job)
            resume_case = output.exists()
            command = worker_command(args, job, slot, resume_case)
            stdout_path = logs / f"{job.mode}--{job.case_id}.stdout.log"
            stderr_path = logs / f"{job.mode}--{job.case_id}.stderr.log"
            stdout = stdout_path.open("ab")
            stderr = stderr_path.open("ab")
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
            stdout.close()
            stderr.close()
            states[job.key] = {
                "status": "running",
                "output": str(output),
                "resumed": resume_case,
                "pid": process.pid,
                "cpu_set": affinity_sets(args)[slot],
                "command": command,
                "started_unix_seconds": int(time.time()),
            }
            running[job.key] = (process, job, slot)

        for key, (process, job, _slot) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            state = states[key]
            state["finished_unix_seconds"] = int(time.time())
            state["return_code"] = return_code
            state["status"] = "complete" if return_code == 0 and is_complete(case_output(args.output, job)) else "failed"
            del running[key]

        if stopping:
            for process, job, _slot in running.values():
                process.terminate()
                states[job.key]["status"] = "interrupted"
            for process, _job, _slot in running.values():
                process.wait(timeout=30)
            atomic_json(args.output / "launcher-progress.json", progress_snapshot(args, plan, states))
            return 130

        atomic_json(args.output / "launcher-progress.json", progress_snapshot(args, plan, states))
        if pending or running:
            time.sleep(args.poll_seconds)

    final = progress_snapshot(args, plan, states)
    atomic_json(args.output / "launcher-progress.json", final)
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
