#!/usr/bin/env python3
"""Run one bounded AQ4 P2 case through an argv-only worker adapter.

Only CPU/synthetic runs are suitable while the parent P1 gate is open.  The
production path remains fail-closed unless every real artifact is present.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 4 * 1024 * 1024
DEFAULT_CAPTURE_BYTES = 256 * 1024
REQUIRED_PREFLIGHT = ("weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes", "gpu_process_snapshot")


class RunnerError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise RunnerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise RunnerError(f"{label} must be a bounded regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(RunnerError(f"non-finite number: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError, RunnerError) as error:
        raise RunnerError(f"cannot parse {label}: {error}") from error


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"{label} must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RunnerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
    with temporary.open("xb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def validate_preflight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError("preflight must be an object")
    for field in REQUIRED_PREFLIGHT:
        if field not in value:
            raise RunnerError(f"preflight field is missing: {field}")
    for field in REQUIRED_PREFLIGHT[:-1]:
        number = value[field]
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise RunnerError(f"preflight.{field} must be a non-negative integer")
    if not isinstance(value["gpu_process_snapshot"], list):
        raise RunnerError("preflight.gpu_process_snapshot must be an array")
    return value


def classify(returncode: int | None, timed_out: bool, overflow: bool) -> str:
    if timed_out:
        return "failed"
    if returncode in (137, -9) or (returncode is not None and returncode == 9):
        return "oom"
    if returncode == 2:
        return "unsupported"
    if returncode == 3:
        return "skipped"
    if overflow:
        return "failed"
    return "ok" if returncode == 0 else "failed"


def capture_process(command: list[str], timeout: float, cap: int) -> tuple[int | None, bytes, bytes, bool, bool, float]:
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, shell=False)
    except OSError:
        return None, b"", b"", False, False, (time.monotonic() - started) * 1000.0
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = False
    timed_out = False
    deadline = started + timeout
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            process.kill()
            remaining = 0.1
        for key, _ in selector.select(min(0.1, max(remaining, 0.01))):
            data = key.fileobj.read1(8192) if hasattr(key.fileobj, "read1") else key.fileobj.read(8192)
            if not data:
                selector.unregister(key.fileobj); key.fileobj.close(); continue
            bucket = buffers[key.data]
            if len(bucket) < cap:
                bucket.extend(data[: cap - len(bucket)])
            if len(data) > max(cap - len(bucket), 0):
                overflow = True
        if process.poll() is not None and not selector.get_map():
            break
    try:
        returncode = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill(); returncode = process.wait()
    return returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]), timed_out, overflow, (time.monotonic() - started) * 1000.0


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    case = load_json(args.case, "case")
    identity = load_json(args.identity, "identity") if args.identity else None
    preflight = validate_preflight(load_json(args.preflight, "preflight"))
    mode = args.mode
    device_id = args.device_id or case.get("device", {}).get("device_id", "cpu-reference")
    trace_sha = None
    oracle_sha = None
    failure_reason = None
    if (mode == "production" or device_id == "r9700-rdna4") and preflight["vram_headroom_bytes"] <= 0:
        failure_reason = "vram_headroom_nonpositive"
    if args.trace:
        if args.trace.is_symlink() or not args.trace.is_file():
            failure_reason = "trace_unavailable"
        else:
            trace_sha = sha_file(args.trace, "trace")
    if args.oracle:
        if args.oracle.is_symlink() or not args.oracle.is_file():
            failure_reason = failure_reason or "source_oracle_unavailable"
        else:
            oracle_sha = sha_file(args.oracle, "source oracle")
    if mode == "production":
        if not isinstance(identity, dict) or identity.get("status") != "bound":
            failure_reason = failure_reason or "identity_not_bound"
        if not args.executable or args.executable.is_symlink() or not args.executable.is_file() or not os.access(args.executable, os.X_OK):
            failure_reason = failure_reason or "production_binary_unavailable"
        if not args.package_root or args.package_root.is_symlink() or not args.package_root.is_dir():
            failure_reason = failure_reason or "production_package_unavailable"
        if not args.trace or trace_sha is None:
            failure_reason = failure_reason or "production_trace_required"
        if not args.oracle or oracle_sha is None:
            failure_reason = failure_reason or "source_oracle_required"
    lock_handle = None
    lock_busy = False
    if device_id == "r9700-rdna4" or args.require_lock:
        if not args.lock:
            failure_reason = failure_reason or "exclusive_lock_required"
        else:
            args.lock.parent.mkdir(parents=True, exist_ok=True)
            lock_handle = args.lock.open("a+")
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_busy = True; failure_reason = failure_reason or "r9700_queue_busy"
    command = list(args.command or [])
    started_at = time.time()
    if failure_reason:
        status = "skipped" if failure_reason in {"r9700_queue_busy", "identity_not_bound"} else "failed"
        returncode, stdout, stderr, elapsed = None, b"", failure_reason.encode(), 0.0
        timed_out = overflow = False
    elif not command:
        status, returncode, stdout, stderr, elapsed, timed_out, overflow = "failed", None, b"", b"command_required", 0.0, False, False
    else:
        returncode, stdout, stderr, timed_out, overflow, elapsed = capture_process(command, args.timeout, args.max_output_bytes)
        status = classify(returncode, timed_out, overflow)
    if lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()
    result = {
        "schema_version": "ullm.aq4_production_p2_raw_result.v1", "case_id": case.get("case_id"), "mode": mode, "device_id": device_id,
        "status": status, "started_at_unix": started_at, "finished_at_unix": time.time(), "elapsed_ms": elapsed,
        "command_argv": command, "returncode": returncode, "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "stdout_truncated": overflow, "preflight": preflight,
        "trace": {"path": str(args.trace), "sha256": trace_sha} if args.trace else None, "source_oracle": {"path": str(args.oracle), "sha256": oracle_sha} if args.oracle else None,
        "failure_reason": failure_reason, "immutable_status": status in {"oom", "failed", "unsupported", "skipped"}, "capture_contract": {"bounded_streaming": True, "shell": False, "max_output_bytes": args.max_output_bytes},
    }
    return result, 0 if status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True); parser.add_argument("--identity", type=Path); parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_synthetic", "production"), default="cpu_synthetic"); parser.add_argument("--device-id")
    parser.add_argument("--trace", type=Path); parser.add_argument("--oracle", type=Path); parser.add_argument("--executable", type=Path); parser.add_argument("--package-root", type=Path)
    parser.add_argument("--lock", type=Path); parser.add_argument("--require-lock", action="store_true"); parser.add_argument("--timeout", type=float, default=300.0); parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_CAPTURE_BYTES)
    # Keep the command as an opaque argv tail.  ``argparse.REMAINDER`` is
    # required because worker arguments such as ``-c`` must never be parsed as
    # runner options.
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.max_output_bytes <= 0 or args.max_output_bytes > MAX_JSON_BYTES: parser.error("--max-output-bytes is out of range")
    try:
        result, code = run(args); atomic_write(args.output, result); print(json.dumps({"status": result["status"], "case_id": result.get("case_id")}, sort_keys=True)); return code
    except (RunnerError, OSError, ValueError) as error:
        print(f"P2 case run failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
