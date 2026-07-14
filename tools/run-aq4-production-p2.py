#!/usr/bin/env python3
"""Run one declared AQ4 P2 worker with bounded, immutable raw evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 16 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 256 * 1024
PREFLIGHT_FIELDS = {"weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes", "gpu_process_snapshot"}


class RunnerError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value: raise RunnerError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES: raise RunnerError(f"{label} must be a bounded regular file")
    try: value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(RunnerError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise RunnerError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise RunnerError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file(): raise RunnerError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    if root.is_symlink() or not root.is_dir(): raise RunnerError("package root is invalid")
    paths = []
    for item in root.rglob("*"):
        if item.is_symlink(): raise RunnerError("package contains a symlink")
        if item.is_file(): paths.append(item)
    if not paths: raise RunnerError("package is empty")
    digest = hashlib.sha256()
    for item in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        digest.update(item.relative_to(root).as_posix().encode()); digest.update(b"\0"); digest.update(bytes.fromhex(sha_file(item, "package file"))); digest.update(b"\n")
    return digest.hexdigest()


def contained(root: Path, path: Path, label: str, *, existing: bool = True) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=existing)
    if resolved != root and root not in resolved.parents: raise RunnerError(f"{label} escapes run root")
    return resolved


def policy_hash(policy: dict[str, Any]) -> str:
    value = json.loads(json.dumps(policy)); value.setdefault("hash_binding", {})["policy_sha256"] = None
    return sha_bytes(canonical(value))


def identity_hash(identity: dict[str, Any]) -> str:
    value = json.loads(json.dumps(identity)); value["identity_sha256"] = None
    return sha_bytes(canonical(value))


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case)); value["case_sha256"] = None
    return sha_bytes(canonical(value))


def validate_preflight(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != PREFLIGHT_FIELDS: raise RunnerError("preflight fields differ")
    for field in PREFLIGHT_FIELDS - {"gpu_process_snapshot"}:
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0: raise RunnerError(f"preflight.{field} must be a non-negative integer")
    snapshot = value["gpu_process_snapshot"]
    if not isinstance(snapshot, list): raise RunnerError("gpu_process_snapshot must be an array")
    normalized = []
    for entry in snapshot:
        if not isinstance(entry, dict) or set(entry) != {"pid", "process_name", "vram_bytes"}: raise RunnerError("GPU process snapshot entry differs")
        if not isinstance(entry["pid"], int) or entry["pid"] <= 0 or not isinstance(entry["process_name"], str) or not entry["process_name"] or not isinstance(entry["vram_bytes"], int) or entry["vram_bytes"] < 0: raise RunnerError("GPU process snapshot entry is invalid")
        normalized.append(entry)
    value["gpu_process_snapshot"] = sorted(normalized, key=lambda item: (item["pid"], item["process_name"]))
    return value


def capture(argv: list[str], timeout: float, limit: int) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    assert process.stdout and process.stderr
    selector = selectors.DefaultSelector(); selector.register(process.stdout, selectors.EVENT_READ, "stdout"); selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    digests = {"stdout": hashlib.sha256(), "stderr": hashlib.sha256()}; counts = {"stdout": 0, "stderr": 0}; timed_out = False
    deadline = started + timeout
    while selector.get_map():
        if time.monotonic() >= deadline and process.poll() is None:
            timed_out = True; process.kill()
        for key, _ in selector.select(0.05):
            chunk = os.read(key.fileobj.fileno(), 64 * 1024)
            if not chunk:
                selector.unregister(key.fileobj); key.fileobj.close(); continue
            digests[key.data].update(chunk); counts[key.data] += len(chunk)
    returncode = process.wait()
    overflow = any(value > limit for value in counts.values())
    status = "failed" if timed_out or overflow else "oom" if returncode in {137, -9} else "unsupported" if returncode == 2 else "skipped" if returncode == 3 else "ok" if returncode == 0 else "failed"
    return {"status": status, "returncode": returncode, "elapsed_ms": (time.monotonic() - started) * 1000.0, "timed_out": timed_out, "output_overflow": overflow, "stdout_sha256": digests["stdout"].hexdigest(), "stderr_sha256": digests["stderr"].hexdigest(), "stdout_bytes": counts["stdout"], "stderr_bytes": counts["stderr"]}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = args.run_root.resolve(strict=True)
    for path, label in ((args.case, "case"), (args.expanded, "expanded"), (args.identity, "identity"), (args.policy, "policy"), (args.preflight, "preflight"), (args.measurement, "measurement"), (args.state, "state"), (args.executable, "executable")):
        contained(root, path, label)
    contained(root, args.package_root, "package root")
    contained(root, args.output, "output", existing=False)
    case = load(args.case, "case"); expanded = load(args.expanded, "expanded"); identity = load(args.identity, "identity"); policy = load(args.policy, "policy")
    preflight = validate_preflight(load(args.preflight, "preflight"))
    load(args.measurement, "measurement"); load(args.state, "state")
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2" or sha_file(args.expanded, "expanded") != identity.get("expanded_manifest_sha256"): raise RunnerError("expanded identity differs")
    matching = [item for item in expanded.get("cases", []) if isinstance(item, dict) and item.get("case_id") == case.get("case_id")]
    if len(matching) != 1 or matching[0] != case or case.get("case_sha256") != case_hash(case): raise RunnerError("case is not the exact expanded case")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("identity_sha256") != identity_hash(identity): raise RunnerError("identity self-binding differs")
    if policy.get("status") != "bound" or policy.get("hash_binding", {}).get("policy_sha256") != policy_hash(policy) or identity.get("policy_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): raise RunnerError("bound policy self-binding differs")
    if str(args.policy.resolve(strict=True)) != identity.get("artifacts", {}).get("bound_policy"): raise RunnerError("bound policy path differs from identity")
    executable = args.executable.resolve(strict=True); package_root = args.package_root.resolve(strict=True)
    if str(executable) != identity.get("artifacts", {}).get("worker") or sha_file(executable, "worker") != identity.get("hash_binding", {}).get("worker_binary_sha256"): raise RunnerError("executable differs from declared worker")
    if str(package_root) != identity.get("artifacts", {}).get("package_root") or tree_hash(package_root) != identity.get("hash_binding", {}).get("package_content_sha256"): raise RunnerError("package differs from declared identity")
    package_manifest = Path(identity.get("artifacts", {}).get("package_manifest", ""))
    contained(root, package_manifest, "package manifest")
    if sha_file(package_manifest, "package manifest") != identity.get("hash_binding", {}).get("package_manifest_sha256"): raise RunnerError("package manifest differs from declared identity")
    argv = [str(executable)]
    if identity.get("execution_contract", {}).get("worker_argv") != argv: raise RunnerError("worker argv contract differs")
    device_id = case.get("device", {}).get("device_id")
    failure_reason = None; lock_handle = None
    if args.mode == "production" and (case.get("scope") != "production_server" or args.trace is None): failure_reason = "production_scope_or_trace_missing"
    if args.trace is not None: contained(root, args.trace, "trace")
    if device_id == "r9700-rdna4":
        lock_name = identity.get("execution_contract", {}).get("r9700_lock_name")
        if args.lock is None or args.lock.name != f"{lock_name}.lock": failure_reason = failure_reason or "canonical_lock_required"
        else:
            contained(root, args.lock, "lock", existing=False); args.lock.parent.mkdir(parents=True, exist_ok=True); lock_handle = args.lock.open("a+")
            try: fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: failure_reason = failure_reason or "r9700_queue_busy"
        allowed = set(identity.get("execution_contract", {}).get("allowed_positive_vram_processes", []))
        if any(entry["vram_bytes"] > 0 and entry["process_name"] not in allowed for entry in preflight["gpu_process_snapshot"]): failure_reason = failure_reason or "foreign_gpu_process"
        minimum = policy.get("power_condition", {}).get("minimum_vram_headroom_bytes")
        if not isinstance(minimum, (int, float)) or preflight["vram_headroom_bytes"] < minimum: failure_reason = failure_reason or "insufficient_vram_headroom"
    started = time.time()
    if failure_reason:
        execution = {"status": "skipped", "returncode": None, "elapsed_ms": 0.0, "timed_out": False, "output_overflow": False, "stdout_sha256": sha_bytes(b""), "stderr_sha256": sha_bytes(failure_reason.encode()), "stdout_bytes": 0, "stderr_bytes": 0}
    else:
        execution = capture(argv, args.timeout, args.max_output_bytes)
    if lock_handle:
        try: fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally: lock_handle.close()
    status = execution["status"]
    raw = {
        "schema_version": "ullm.aq4_production_p2_raw_result.v2", "case_id": case["case_id"], "case_sha256": case["case_sha256"],
        "status": status, "immutable_status": status != "ok", "mode": args.mode, "device_id": device_id,
        "started_at_unix": started, "finished_at_unix": time.time(), "execution": execution,
        "declared_execution": {"executable": str(executable), "executable_sha256": sha_file(executable, "worker"), "package_root": str(package_root), "package_content_sha256": tree_hash(package_root), "argv_sha256": sha_bytes(canonical(argv)), "argv_count": len(argv), "argv_values_recorded": False},
        "links": {"expanded": {"path": str(args.expanded.resolve()), "sha256": sha_file(args.expanded, "expanded")}, "identity": {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity")}, "policy": {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy")}, "measurement": {"path": str(args.measurement.resolve()), "sha256": sha_file(args.measurement, "measurement")}, "state": {"path": str(args.state.resolve()), "sha256": sha_file(args.state, "state")}, "trace": {"path": str(args.trace.resolve()), "sha256": sha_file(args.trace, "trace")} if args.trace else None},
        "preflight": preflight, "failure_reason": failure_reason, "capture_contract": {"bounded_streaming": True, "shell": False, "max_output_bytes_per_stream": args.max_output_bytes, "command_arguments_stored": False},
    }
    return raw, 0 if status == "ok" else 1


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise RunnerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "case", "expanded", "identity", "policy", "preflight", "measurement", "state", "executable", "package_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_synthetic", "production"), default="cpu_synthetic"); parser.add_argument("--trace", type=Path); parser.add_argument("--lock", type=Path); parser.add_argument("--timeout", type=float, default=300.0); parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_OUTPUT_LIMIT)
    args = parser.parse_args(argv)
    if args.max_output_bytes <= 0 or args.max_output_bytes > 64 * 1024 * 1024: parser.error("--max-output-bytes is out of range")
    try:
        raw, code = run(args); atomic_write(args.output, raw); print(json.dumps({"status": raw["status"], "case_id": raw["case_id"]}, sort_keys=True)); return code
    except (RunnerError, OSError, ValueError) as error:
        print(f"P2 case run failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
