#!/usr/bin/env python3
"""Run one bounded direct-engine Phase 5 row and retain only local raw data.

The executable, model lock/cache, and output directory are all explicit.  A
successful manifest proves the process boundary, exact row identity, clean
health before/after, no fallback/interference, and one model load reused by
the ten measured requests.  Raw CLI JSON remains under the caller's local
artifact directory; the manifest stores its digest only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, IO, Mapping, NoReturn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
from engine_performance_common import (  # noqa: E402
    CLAIMS,
    BUILD_IDENTITY_VERSION,
    DIRECT_SCHEMA_PATH,
    DIRECT_VERSION,
    EVIDENCE_VERSION,
    MATRIX_PATH,
    ROCM_RELEASE,
    ROCM_ROOT,
    cache_digest,
    expected_build_configuration,
    expected_device,
    expected_model,
    load_matrix,
    parse_json_bytes,
    read_json,
    resolved_row,
    schema_validate,
    sha256_file,
    validate_build_configuration,
    validate_cli_result,
)


MAX_BINARY_BYTES = 4 * 1024 * 1024 * 1024
MAX_RAW_BYTES = 64 * 1024 * 1024
AMD_SMI_EXECUTABLE = "/opt/rocm/core-7.14/bin/amd-smi"
TERMINATION_GRACE_SECONDS = 2
MONITOR_CADENCE_SECONDS = 1.0
MONITOR_ACQUISITION_TIMEOUT_SECONDS = 30.0
MONITOR_ACQUISITION_POLL_SECONDS = 0.1
MONITOR_COMMAND_TIMEOUT_SECONDS = 8
METRIC_TELEMETRY_ATTEMPTS = 3
METRIC_TELEMETRY_RETRY_SECONDS = 0.1
MAX_MONITOR_SAMPLES = 100_000
MAX_LOADER_PATHS = 128
MAX_LIBRARY_BYTES = 512 * 1024 * 1024
PIPE_READ_BYTES = 64 * 1024
PIPE_SPOOL_MEMORY_BYTES = 1024 * 1024
PROCESS_POLL_SECONDS = 0.05
VISIBILITY_NAMES = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
ROCM_LIBRARY_NAMES = ("libamdhip64.so", "libhsa-runtime64.so")
ROCM_LIBRARY_PATH = f"{ROCM_ROOT}/lib"
PHASE5_ALLOWED_TARGET_PIDS_ENV = "SLLM_PHASE5_ALLOWED_TARGET_PIDS"
MAX_ALLOWED_TARGET_VRAM_BYTES = 1 * 1024 * 1024
MAX_ALLOWED_TARGET_GTT_BYTES = 16 * 1024 * 1024
MAX_LINUX_PID = (1 << 31) - 1


def _fail(message: str) -> NoReturn:
    raise ContractError(message)


class MonitorNotReady(ContractError):
    """The owned process exists but GPU context/ROCm loader evidence is not ready yet."""


def _write(path: Path, data: bytes, label: str) -> None:
    try:
        if path.exists() or path.is_symlink():
            _fail(f"refusing to overwrite existing {label}: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        _fail(f"cannot write {label} {path}: {exc}")


def _regular_executable(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        _fail(f"benchmark binary must be an executable regular file: {path}")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        _fail(f"benchmark binary cannot be resolved: {exc}")
    if path.stat().st_size > MAX_BINARY_BYTES:
        _fail(f"benchmark binary exceeds bounded size: {path}")
    return path


def _regular_file(path: Path, label: str, max_bytes: int = MAX_RAW_BYTES) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size > max_bytes:
        _fail(f"{label} exceeds bounded size: {path}")
    return path


def _git_output(source_root: Path, arguments: list[str], label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            capture_output=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"cannot validate build identity {label}: {exc}")
    if completed.returncode != 0 or completed.stderr:
        _fail(f"build identity {label} is not available in the selected repository")
    return completed.stdout.decode("ascii", errors="strict").strip()


def _validate_source_base(source_root: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        _fail("build identity source base revision is not an immutable commit")
    if _git_output(source_root, ["cat-file", "-t", revision], "source base revision") != "commit":
        _fail("build identity source base revision is not a commit object")
    head = _git_output(source_root, ["rev-parse", "HEAD"], "HEAD")
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "merge-base", "--is-ancestor", revision, head],
            capture_output=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"cannot validate build identity source ancestry: {exc}")
    if completed.returncode != 0:
        _fail("build identity source base revision is not an ancestor/base identity of HEAD")


def _validate_semantic_tree(source_root: Path, tree: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        _fail("build identity semantic tree is malformed")
    if _git_output(source_root, ["cat-file", "-t", tree], "semantic tree") != "tree":
        _fail("build identity semantic tree is not a Git tree object")


def _validate_build_manifest(path: Path, binary: Path, target: str, repo: Path) -> tuple[dict[str, Any], str]:
    path = _regular_file(path.resolve(), "build identity manifest", 4 * 1024 * 1024)
    document, raw, digest = read_json(path, "build identity manifest", 4 * 1024 * 1024)
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "source_root", "source_base_revision", "semantic_tree", "build_inputs_digest",
        "build_configuration", "target", "backend", "rocm_release", "rocm_root", "binary_sha256",
    }:
        _fail("build identity manifest is incomplete or has unexpected fields")
    if document["schema_version"] != BUILD_IDENTITY_VERSION:
        _fail("build identity manifest version is stale")
    source_root = _resolve_repo_path(str(document["source_root"]), repo, "build identity source root")
    if not source_root.is_dir():
        _fail("build identity source root is not a directory")
    _validate_source_base(source_root, str(document["source_base_revision"]))
    _validate_semantic_tree(source_root, str(document["semantic_tree"]))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(document["build_inputs_digest"])):
        _fail("build identity build-input digest is malformed")
    if document["target"] != target or document["backend"] != "hip":
        _fail("build identity target/backend does not match the row")
    validate_build_configuration(document["build_configuration"], target, "build identity build configuration")
    if document["rocm_release"] != ROCM_RELEASE or document["rocm_root"] != ROCM_ROOT:
        _fail("build identity ROCm tuple is not the required release/root")
    binary_digest = sha256_file(binary, "benchmark binary", max_bytes=MAX_BINARY_BYTES)
    if document["binary_sha256"] != binary_digest:
        _fail("build identity does not bind the selected benchmark binary")
    return document, digest


def _resolve_repo_path(path_value: str, repo: Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label} {path}: {exc}")


def _validate_lock(lock_path: Path, model_size: str, repo: Path) -> tuple[dict[str, Any], str]:
    expected = expected_model(model_size)
    expected_path = (repo / expected["lock_path"]).resolve()
    if lock_path.resolve() != expected_path:
        _fail("model lock path is not the exact matrix-bound lock")
    lock, _, digest = read_json(lock_path, "model lock", 16 * 1024 * 1024)
    if not isinstance(lock, dict) or not isinstance(lock.get("model"), dict):
        _fail("model lock has no model identity")
    model = lock["model"]
    if model.get("repo_id") != expected["repo_id"] or model.get("resolved_revision") != expected["resolved_revision"]:
        _fail("model lock model/revision identity is stale")
    if lock.get("fingerprint") != expected["lock_fingerprint"]:
        _fail("model lock fingerprint is stale or wrong")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        _fail("model lock has no bounded file list")
    for item in files:
        if not isinstance(item, dict) or set(item) < {"path", "size_bytes", "sha256"}:
            _fail("model lock file entry is incomplete")
        if not isinstance(item["path"], str) or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
            _fail("model lock contains an unsafe cache path")
    return lock, digest


def _validate_cache(cache_path: Path, lock: Mapping[str, Any]) -> str:
    digest = cache_digest(cache_path)
    expected: dict[str, tuple[int, str]] = {}
    for item in lock["model"]["files"]:
        expected[item["path"]] = (item["size_bytes"], item["sha256"])
    observed: set[str] = set()
    for path_value, (size_bytes, expected_sha) in expected.items():
        path = cache_path / path_value
        observed.add(path_value)
        if path.is_symlink() or not path.is_file():
            _fail(f"model cache is missing locked file: {path_value}")
        if path.stat().st_size != size_bytes or sha256_file(path, f"model cache {path_value}") != expected_sha:
            _fail(f"model cache file is stale or tampered: {path_value}")
    actual_files = {item.relative_to(cache_path).as_posix() for item in cache_path.rglob("*") if item.is_file()}
    if actual_files != observed:
        _fail("model cache file set differs from the model lock")
    return digest


def _expected_command(binary: Path, row: Mapping[str, Any], lock: Path, cache: Path) -> list[str]:
    row = resolved_row(row)
    return [
        str(binary), "benchmark", "--lane", "direct", "--lock", str(lock), "--cache", str(cache),
        "--row-id", row["row_id"], "--model-size", row["model_size"], "--case-id", row["case_id"],
        "--input-token-ids", ",".join(str(token) for token in row["input_token_ids"]),
        "--max-new-tokens", str(row["requested_output_tokens"]), "--device-index", "0",
        "--target", row["target"], "--greedy", "--warmups", "3", "--measured", "10",
    ]


def _execution_environment(row_id: str, target: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(base if base is not None else os.environ)
    for name in VISIBILITY_NAMES:
        environment.pop(name, None)
    environment.pop(PHASE5_ALLOWED_TARGET_PIDS_ENV, None)
    environment["ROCR_VISIBLE_DEVICES"] = expected_device(target)["gpu_uuid"]
    environment["LD_LIBRARY_PATH"] = ROCM_LIBRARY_PATH
    environment["SLLM_ENGINE_PERFORMANCE_ROW"] = row_id
    return environment


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _send_group(pid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _wait_process_group_gone(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _process_group_gone(pid):
            return True
        time.sleep(0.02)
    return _process_group_gone(pid)


def _bounded_process_output(
    process: subprocess.Popen[bytes], timeout_seconds: float,
) -> tuple[bytes, bytes, bool, bool, bool, list[str]]:
    """Drain both child pipes concurrently while enforcing the raw-output bound."""
    if process.stdout is None or process.stderr is None:
        _fail("benchmark process pipes were not created")
    pipes: dict[str, IO[bytes]] = {"stdout": process.stdout, "stderr": process.stderr}
    sizes = {name: 0 for name in pipes}
    overflow: set[str] = set()
    timed_out = False
    term_sent = False
    kill_sent = False
    term_started: float | None = None
    kill_started: float | None = None
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))

    def request_term(now: float) -> None:
        nonlocal term_sent, term_started
        if term_started is None:
            term_sent = _send_group(process.pid, signal.SIGTERM) or term_sent
            term_started = now

    def request_kill(now: float) -> None:
        nonlocal kill_sent, kill_started
        if kill_started is None:
            kill_sent = _send_group(process.pid, signal.SIGKILL) or kill_sent
            kill_started = now

    with tempfile.SpooledTemporaryFile(
        max_size=PIPE_SPOOL_MEMORY_BYTES, mode="w+b",
    ) as stdout_spool, tempfile.SpooledTemporaryFile(
        max_size=PIPE_SPOOL_MEMORY_BYTES, mode="w+b",
    ) as stderr_spool:
        spools = {"stdout": stdout_spool, "stderr": stderr_spool}
        selector = selectors.DefaultSelector()
        try:
            for name, pipe in pipes.items():
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while True:
                now = time.monotonic()
                returncode = process.poll()
                group_gone = _process_group_gone(process.pid)
                capture_complete = returncode is not None and group_gone and not selector.get_map()
                if term_started is None and now >= deadline and not capture_complete:
                    timed_out = True
                    request_term(now)
                if term_started is None and returncode is not None and not group_gone:
                    request_term(now)
                if term_started is not None and now >= term_started + TERMINATION_GRACE_SECONDS:
                    if not group_gone:
                        request_kill(now)
                    elif returncode is not None and selector.get_map():
                        break
                if kill_started is not None and now >= kill_started + TERMINATION_GRACE_SECONDS:
                    break
                if capture_complete:
                    break

                wake_at = now + PROCESS_POLL_SECONDS
                if term_started is None:
                    wake_at = min(wake_at, deadline)
                elif kill_started is None:
                    wake_at = min(wake_at, term_started + TERMINATION_GRACE_SECONDS)
                else:
                    wake_at = min(wake_at, kill_started + TERMINATION_GRACE_SECONDS)
                events = selector.select(max(0.0, wake_at - now))
                for key, _mask in events:
                    stream_name: str = key.data
                    pipe = pipes[stream_name]
                    try:
                        chunk = os.read(pipe.fileno(), PIPE_READ_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        pipe.close()
                        continue
                    remaining = MAX_RAW_BYTES - sizes[stream_name]
                    if remaining > 0:
                        retained = chunk[:remaining]
                        spools[stream_name].write(retained)
                        sizes[stream_name] += len(retained)
                    if len(chunk) > remaining:
                        overflow.add(stream_name)
                        request_term(time.monotonic())
            process.poll()
        finally:
            selector.close()
            for pipe in pipes.values():
                if not pipe.closed:
                    pipe.close()

        output: dict[str, bytes] = {}
        for name, spool in spools.items():
            spool.seek(0)
            output[name] = spool.read(MAX_RAW_BYTES)
        return output["stdout"], output["stderr"], timed_out, term_sent, kill_sent, sorted(overflow)


def _execute_bounded(
    command: list[str], environment: Mapping[str, str], cwd: Path, timeout_seconds: float,
    *, monitor_provider: Callable[[str, int, int], dict[str, Any]] | None = None,
    monitor_target: str | None = None,
) -> dict[str, Any]:
    if monitor_provider is not None and monitor_target is None:
        _fail("a monitor target is required when monitoring is enabled")
    child_environment = dict(environment)
    child_environment.pop(PHASE5_ALLOWED_TARGET_PIDS_ENV, None)
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=child_environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
        )
    except OSError as exc:
        _fail(f"cannot start benchmark binary: {exc}")
    timed_out = False
    term_sent = False
    kill_sent = False
    output_overflow: list[str] = []
    stdout = b""
    stderr = b""
    monitor_stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    monitor_result: dict[str, Any] = {"samples": [], "errors": []}
    if monitor_provider is not None:
        assert monitor_target is not None
        monitor_thread = threading.Thread(
            target=lambda: monitor_result.update(_monitor_loop(monitor_target, process.pid, monitor_provider, monitor_stop)),
            name="engine-performance-monitor", daemon=True,
        )
        monitor_thread.start()
    try:
        stdout, stderr, timed_out, term_sent, kill_sent, output_overflow = _bounded_process_output(
            process, timeout_seconds,
        )
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=(MONITOR_COMMAND_TIMEOUT_SECONDS * 4) + TERMINATION_GRACE_SECONDS)
            if monitor_thread.is_alive():
                monitor_result["errors"].append("monitor worker did not terminate within its bounded teardown window")
        if not _process_group_gone(process.pid):
            term_sent = _send_group(process.pid, signal.SIGTERM) or term_sent
            if not _wait_process_group_gone(process.pid, TERMINATION_GRACE_SECONDS):
                kill_sent = _send_group(process.pid, signal.SIGKILL) or kill_sent
                _wait_process_group_gone(process.pid, TERMINATION_GRACE_SECONDS)
        process.poll()
    duration_ns = time.monotonic_ns() - started
    return {
        "stdout": stdout if isinstance(stdout, bytes) else bytes(stdout),
        "stderr": stderr if isinstance(stderr, bytes) else bytes(stderr),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "duration_ns": duration_ns,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "process_group_gone": _process_group_gone(process.pid),
        "output_overflow": output_overflow,
        "monitor": monitor_result,
    }


def _run_json_command(command: list[str], timeout: int = MONITOR_COMMAND_TIMEOUT_SECONDS) -> Any:
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"health command failed: {exc}")
    if completed.returncode != 0 or completed.stderr != b"":  # type: ignore[union-attr]
        _fail("health command did not exit cleanly")
    return parse_json_bytes(completed.stdout, "health command")  # type: ignore[union-attr]


def _child_process_ids(parent_pid: int) -> list[int]:
    children: list[int] = []
    try:
        for item in Path("/proc").iterdir():
            if not item.name.isdigit():
                continue
            try:
                fields = (item / "stat").read_text(encoding="ascii").split()
                if int(fields[3]) == parent_pid:
                    children.append(int(fields[0]))
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        return [-1]
    return sorted(children)


def _proc_relationship(pid: int) -> tuple[int, int] | None:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        after_comm = text.rsplit(")", 1)[1].strip().split()
        return int(after_comm[1]), int(after_comm[2])
    except (OSError, ValueError, IndexError):
        return None


def _process_group_members(root_pid: int) -> set[int]:
    root = _proc_relationship(root_pid)
    if root is None:
        return set()
    root_pgrp = root[1]
    records: dict[int, tuple[int, int]] = {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return set()
    for item in proc_entries:
        if not item.name.isdigit():
            continue
        relationship = _proc_relationship(int(item.name))
        if relationship is not None:
            records[int(item.name)] = relationship
    members = {pid for pid, (_ppid, pgrp) in records.items() if pgrp == root_pgrp}
    members.add(root_pid)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _pgrp) in records.items():
            if ppid in members and pid not in members:
                members.add(pid)
                changed = True
    return members


def _proc_maps_paths(pids: set[int]) -> tuple[set[str], set[int]]:
    paths: set[str] = set()
    readable: set[int] = set()
    for pid in sorted(pids):
        try:
            lines = (Path("/proc") / str(pid) / "maps").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        readable.add(pid)
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            path = fields[5].strip()
            if path.endswith(" (deleted)"):
                path = path[:-10]
            if path.startswith("/") and ("/rocm" in path.lower() or any(name in Path(path).name for name in ROCM_LIBRARY_NAMES)):
                paths.add(path)
    return paths, readable


def _loader_evidence(process_ids: set[int]) -> dict[str, Any]:
    if not process_ids:
        raise MonitorNotReady("ROCm loader acquisition is waiting for a live owned benchmark process")
    paths, readable = _proc_maps_paths(process_ids)
    if not readable:
        raise MonitorNotReady("ROCm loader acquisition is waiting for readable benchmark process maps")
    resolved: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        try:
            canonical = str(path.resolve(strict=True))
        except OSError as exc:
            _fail(f"resolved loader path is unavailable: {path}: {exc}")
        if not canonical.startswith(ROCM_ROOT + "/"):
            _fail(f"benchmark resolved an unexpected ROCm root: {canonical}")
        resolved.add(canonical)
    required = {name: [path for path in resolved if Path(path).name.startswith(name)] for name in ROCM_LIBRARY_NAMES}
    if any(not values for values in required.values()):
        raise MonitorNotReady("ROCm loader acquisition is waiting for both HIP and ROCr runtime libraries")
    if len(resolved) > MAX_LOADER_PATHS:
        _fail("benchmark resolved too many ROCm/HIP loader paths")
    library_digests: dict[str, str] = {}
    for path_value in sorted(resolved):
        library_digests[path_value] = sha256_file(Path(path_value), f"resolved ROCm library {path_value}", max_bytes=MAX_LIBRARY_BYTES)
    path_digest = "sha256:" + hashlib.sha256(canonical_bytes(sorted(resolved))).hexdigest()
    return {
        "required_rocm_release": ROCM_RELEASE,
        "expected_root": ROCM_ROOT,
        "resolved_paths": sorted(resolved),
        "path_digest": path_digest,
        "library_digests": library_digests,
        "process_ids": sorted(readable),
    }


def _extract_process_pids(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    key_lower = key.lower()
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            found.update(_extract_process_pids(child_value, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(_extract_process_pids(child, key))
    elif isinstance(value, int) and not isinstance(value, bool) and "pid" in key_lower and value > 0:
        found.add(value)
    elif isinstance(value, str) and "pid" in key_lower:
        found.update(int(match) for match in re.findall(r"\b(\d+)\b", value))
    elif isinstance(value, str):
        found.update(int(match) for match in re.findall(r"\bpid\s*[:=]?\s*(\d+)\b", value, flags=re.IGNORECASE))
    return found


def _process_records(process_doc: Any, expected_gpu_index: int) -> list[Any]:
    if not isinstance(process_doc, list) or len(process_doc) != 1 or not isinstance(process_doc[0], dict):
        _fail("AMD-SMI process observation is not bound to one device")
    if process_doc[0].get("gpu") != expected_gpu_index:
        _fail("AMD-SMI process observation selected the wrong GPU")
    process_list = process_doc[0].get("process_list")
    if process_list == [{"process_info": "No running processes detected"}]:
        return []
    if not isinstance(process_list, list) or not process_list:
        _fail("AMD-SMI process list is missing or malformed")
    return process_list


def _parse_phase5_allowed_target_pids(environment: Mapping[str, str] | None = None) -> tuple[int, ...]:
    value = (environment if environment is not None else os.environ).get(PHASE5_ALLOWED_TARGET_PIDS_ENV)
    if value is None or value == "":
        return ()
    if not re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", value):
        _fail(f"{PHASE5_ALLOWED_TARGET_PIDS_ENV} must be a strict comma-separated list of positive decimal PIDs")
    pids = tuple(int(item) for item in value.split(","))
    if any(pid > MAX_LINUX_PID for pid in pids):
        _fail(f"{PHASE5_ALLOWED_TARGET_PIDS_ENV} contains a PID outside the supported range")
    if len(set(pids)) != len(pids):
        _fail(f"{PHASE5_ALLOWED_TARGET_PIDS_ENV} contains a duplicate PID")
    return pids


def _typed_nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"AMD-SMI allowed target process {label} is missing or malformed")
    return value


def _typed_process_measurement(value: Any, unit: str, label: str) -> int:
    if not isinstance(value, dict) or set(value) != {"value", "unit"} or value.get("unit") != unit:
        _fail(f"AMD-SMI allowed target process {label} is missing or malformed")
    return _typed_nonnegative_integer(value.get("value"), label)


def _typed_cu_occupancy(value: Any) -> int | str:
    if value == "N/A":
        return value
    return _typed_nonnegative_integer(value, "CU occupancy")


def _process_record_pid(record: Any) -> int:
    if not isinstance(record, dict) or set(record) != {"process_info"} or not isinstance(record["process_info"], dict):
        _fail("AMD-SMI target process record is missing or malformed")
    pid = record["process_info"].get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid > MAX_LINUX_PID:
        _fail("AMD-SMI target process PID is missing or malformed")
    return pid


def _validate_inert_allowed_process_record(record: Any) -> int:
    pid = _process_record_pid(record)
    info = record["process_info"]
    if set(info) != {
        "name", "pid", "memory_usage", "mem_usage", "usage", "sdma_usage", "cu_occupancy", "evicted_time",
    } or not isinstance(info["name"], str) or not info["name"]:
        _fail("AMD-SMI allowed target process record is not fully typed")
    memory = info["memory_usage"]
    usage = info["usage"]
    if not isinstance(memory, dict) or set(memory) != {"gtt_mem", "cpu_mem", "vram_mem"}:
        _fail("AMD-SMI allowed target process memory usage is not fully typed")
    if not isinstance(usage, dict) or set(usage) != {"gfx", "enc"}:
        _fail("AMD-SMI allowed target process engine usage is not fully typed")
    gtt_bytes = _typed_process_measurement(memory["gtt_mem"], "B", "GTT memory")
    _typed_process_measurement(memory["cpu_mem"], "B", "CPU memory")
    vram_bytes = _typed_process_measurement(memory["vram_mem"], "B", "VRAM memory")
    _typed_process_measurement(info["mem_usage"], "B", "memory usage")
    gfx_usage = _typed_process_measurement(usage["gfx"], "ns", "GFX usage")
    _typed_process_measurement(usage["enc"], "ns", "encoder usage")
    _typed_process_measurement(info["sdma_usage"], "us", "SDMA usage")
    _typed_cu_occupancy(info["cu_occupancy"])
    _typed_process_measurement(info["evicted_time"], "ms", "evicted time")
    if vram_bytes > MAX_ALLOWED_TARGET_VRAM_BYTES:
        _fail("AMD-SMI allowed target process exceeds the 1 MiB VRAM boundary")
    if gtt_bytes > MAX_ALLOWED_TARGET_GTT_BYTES:
        _fail("AMD-SMI allowed target process exceeds the 16 MiB GTT boundary")
    if gfx_usage != 0:
        _fail("AMD-SMI allowed target process has nonzero GFX activity")
    return pid


def _allowed_process_observation(process_list: list[Any], allowed_pids: tuple[int, ...]) -> list[dict[str, Any]]:
    allowed = set(allowed_pids)
    seen: set[int] = set()
    for record in process_list:
        pid = _process_record_pid(record)
        if pid in seen:
            _fail("AMD-SMI target process observation contains a duplicate PID")
        seen.add(pid)
        if pid not in allowed:
            _fail("AMD-SMI target process observation contains an unallowlisted PID")
        _validate_inert_allowed_process_record(record)
    if not allowed_pids:
        return []
    audit: list[dict[str, Any]] = [{"allowlisted_pids": list(allowed_pids)}]
    audit.extend({"record": record, "record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()} for record in process_list)
    return audit


def _partition_during_processes(process_list: list[Any], allowed_pids: tuple[int, ...]) -> list[Any]:
    allowed = set(allowed_pids)
    seen: set[int] = set()
    owned: list[Any] = []
    for record in process_list:
        pid = _process_record_pid(record)
        if pid in seen:
            _fail("during-run AMD-SMI process observation contains a duplicate PID")
        seen.add(pid)
        if pid in allowed:
            _validate_inert_allowed_process_record(record)
        else:
            owned.append(record)
    return owned


def _number(value: Any, unit: str, label: str) -> float:
    if not isinstance(value, dict) or set(value) != {"value", "unit"} or not isinstance(value["value"], (int, float)) or isinstance(value["value"], bool) or value["unit"] != unit:
        _fail(f"AMD-SMI {label} is missing or malformed")
    number = float(value["value"])
    if not math.isfinite(number) or number < 0:
        _fail(f"AMD-SMI {label} is outside the bounded range")
    return number


def _throttle_evidence(throttle: Any, power_status: str) -> dict[str, Any]:
    if not isinstance(throttle, dict) or not isinstance(power_status, str) or not power_status:
        _fail("AMD-SMI throttle evidence is missing or malformed")
    values = [value for value in throttle.values() if value != "N/A"]
    if any(not isinstance(value, (str, int, float, bool)) for value in values):
        _fail("AMD-SMI throttle evidence contains an unsupported value")
    # On the canonical RDNA targets the legacy aggregate throttle bit can
    # alternate while the board is idle and all reason/accumulator fields are
    # unavailable.  Treat that bit as observational unless AMD-SMI exposes a
    # reason that can be evaluated.  Temperature and socket-power limits are
    # checked independently for every pre/during/post sample.
    explicit = bool(values) and power_status.upper() != "UNTHROTTLED"
    for value in values:
        normalized = str(value).upper()
        if normalized not in {"NORMAL", "INACTIVE", "NO", "FALSE", "0", "N/A"}:
            if any(token in normalized for token in ("VIOLATION", "THROTTLE", "ACTIVE", "PROCHOT")):
                explicit = True
    available = bool(values)
    reason = "AMD-SMI reports all violation/accumulator fields as N/A on this non-MI300 target" if not available else "AMD-SMI violation fields were exposed and captured"
    return {
        "power_statuses": [power_status], "explicit_violation": explicit,
        "accumulator_available": available, "accumulator_reason": reason,
        "accumulator_digest": "sha256:" + hashlib.sha256(canonical_bytes(throttle)).hexdigest(),
    }


def _amd_smi_version() -> dict[str, str]:
    try:
        completed = subprocess.run([AMD_SMI_EXECUTABLE, "version"], capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"AMD-SMI version query failed: {exc}")
    if completed.returncode != 0 or completed.stderr:
        _fail("AMD-SMI version query did not exit cleanly")
    line = completed.stdout.decode("utf-8", errors="strict").strip().splitlines()[0]
    fields = [part.strip() for part in line.split("|")]
    values: dict[str, str] = {}
    for field in fields:
        if ":" not in field:
            continue
        key, value = field.split(":", 1)
        values[key.strip().lower().replace(" ", "_")] = value.strip()
    required = {"amdsmi_tool", "amdsmi_library_version", "rocm_version"}
    if not required.issubset(values):
        _fail("AMD-SMI version output is malformed")
    if values["rocm_version"] != ROCM_RELEASE:
        _fail("AMD-SMI reports an unexpected ROCm release")
    return {"tool_version": values["amdsmi_tool"], "library_version": values["amdsmi_library_version"], "rocm_version": values["rocm_version"]}


def _amd_smi_list_identity(target: str) -> tuple[dict[str, Any], int]:
    expected = expected_device(target)
    listed = _run_json_command([AMD_SMI_EXECUTABLE, "list", "-e", "--json"])
    if not isinstance(listed, list):
        _fail("AMD-SMI device list is not an array")
    # HIP UUID is the primary identity. BDF and physical HIP index are required
    # corroborating fields because this AMD-SMI release cannot select GPU-* UUIDs.
    matches = [item for item in listed if isinstance(item, dict) and item.get("hip_uuid") == expected["gpu_uuid"]]
    if len(matches) != 1:
        _fail("AMD-SMI UUID-primary device mapping is missing or ambiguous")
    match = matches[0]
    if match.get("bdf", "").lower() != expected["gpu_bdf"] or match.get("hip_id") != expected["physical_hip_index"]:
        _fail("AMD-SMI UUID mapping has target/BDF/HIP-index drift")
    if not isinstance(match.get("gpu"), int):
        _fail("AMD-SMI device mapping has no host GPU index")
    return match, match["gpu"]


def _static_evidence(target: str, match: Mapping[str, Any], amd_smi_gpu_index: int) -> dict[str, Any]:
    expected = expected_device(target)
    document = _run_json_command([AMD_SMI_EXECUTABLE, "static", "-a", "-b", "-d", "-v", "-C", "ALL", "-l", "-o", "--json", "-g", expected["gpu_bdf"]])
    if not isinstance(document, dict) or not isinstance(document.get("gpu_data"), list) or len(document["gpu_data"]) != 1 or not isinstance(document["gpu_data"][0], dict):
        _fail("AMD-SMI static evidence is not bound to one device")
    data = document["gpu_data"][0]
    if data.get("gpu") != amd_smi_gpu_index or data.get("bus", {}).get("bdf", "").lower() != expected["gpu_bdf"]:
        _fail("AMD-SMI static evidence has device drift")
    asic = data.get("asic")
    driver = data.get("driver")
    profile = data.get("profile")
    limits = data.get("limit")
    vram = data.get("vram")
    clocks = data.get("clock")
    if not isinstance(asic, dict) or asic.get("market_name") != expected["product"] or asic.get("target_graphics_version") != target:
        _fail("AMD-SMI static product/target identity is stale")
    if not isinstance(driver, dict) or not isinstance(driver.get("version"), str) or not isinstance(driver.get("os_kernel_version"), str):
        _fail("AMD-SMI driver/kernel evidence is missing")
    if not isinstance(profile, dict) or not isinstance(profile.get("current"), str) or not isinstance(profile.get("available_profiles"), list) or not profile["available_profiles"]:
        _fail("AMD-SMI power profile evidence is malformed")
    if not isinstance(limits, dict) or not isinstance(clocks, dict) or not isinstance(vram, dict):
        _fail("AMD-SMI limits/clock/VRAM evidence is malformed")
    vram_size = _number(vram.get("size"), "MB", "static VRAM size")
    profile_record = {"current": profile["current"], "available_profiles": profile["available_profiles"], "digest": "sha256:" + hashlib.sha256(canonical_bytes(profile)).hexdigest()}
    clock_levels = {name: value for name, value in clocks.items() if isinstance(value, dict) and "frequency_levels" in value}
    clock_record = {"values": clock_levels, "digest": "sha256:" + hashlib.sha256(canonical_bytes(clock_levels)).hexdigest()}
    limit_record = {"values": limits, "digest": "sha256:" + hashlib.sha256(canonical_bytes(limits)).hexdigest()}
    return {
        "target": target, "product": expected["product"], "gpu_bdf": expected["gpu_bdf"], "gpu_uuid": expected["gpu_uuid"],
        "physical_hip_index": expected["physical_hip_index"], "amd_smi_gpu_index": amd_smi_gpu_index,
        "driver_version": driver["version"], "kernel_version": driver["os_kernel_version"],
        "profile": profile_record, "limits": limit_record, "clock_levels": clock_record, "vram_total_mb": vram_size,
    }


def _metric_evidence_once(target: str, amd_smi_gpu_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = expected_device(target)
    document = _run_json_command([AMD_SMI_EXECUTABLE, "metric", "-m", "-u", "-p", "-c", "-t", "-l", "-v", "-e", "--json", "-g", expected["gpu_bdf"]])
    if not isinstance(document, dict) or not isinstance(document.get("gpu_data"), list) or len(document["gpu_data"]) != 1 or not isinstance(document["gpu_data"][0], dict):
        _fail("AMD-SMI metric evidence is not bound to one device")
    data = document["gpu_data"][0]
    if data.get("gpu") != amd_smi_gpu_index:
        _fail("AMD-SMI metric evidence selected the wrong device")
    temperature = data.get("temperature")
    clock = data.get("clock")
    power = data.get("power")
    mem_usage = data.get("mem_usage")
    ecc = data.get("ecc")
    perf_level = data.get("perf_level")
    if not isinstance(temperature, dict) or not isinstance(clock, dict) or not isinstance(power, dict) or not isinstance(mem_usage, dict) or not isinstance(ecc, dict) or not isinstance(perf_level, str):
        _fail("AMD-SMI metric evidence is malformed")
    metric = {
        "temperature_c": {name: _number(temperature.get(name), "C", f"temperature {name}") for name in ("edge", "hotspot", "mem")},
        "gfx_clock_mhz": _number(clock.get("gfx_0", {}).get("clk"), "MHz", "GFX clock"),
        "mem_clock_mhz": _number(clock.get("mem_0", {}).get("clk"), "MHz", "memory clock"),
        "power_w": _number(power.get("socket_power"), "W", "socket power"),
        "perf_level": perf_level,
        "throttle_status": power.get("throttle_status"),
        "vram_used_mb": _number(mem_usage.get("used_vram"), "MB", "metric VRAM used"),
        "vram_total_mb": _number(mem_usage.get("total_vram"), "MB", "metric VRAM total"),
        "ecc_uncorrectable": ecc.get("total_uncorrectable_count"),
    }
    if not isinstance(metric["throttle_status"], str) or not isinstance(metric["ecc_uncorrectable"], int) or isinstance(metric["ecc_uncorrectable"], bool) or metric["ecc_uncorrectable"] < 0:
        _fail("AMD-SMI throttle/ECC metric is malformed")
    metric["metric_digest"] = "sha256:" + hashlib.sha256(canonical_bytes(data)).hexdigest()
    return metric, _throttle_evidence(data.get("throttle"), metric["throttle_status"])


def _metric_evidence(target: str, amd_smi_gpu_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retry a bounded number of transient dynamic-telemetry gaps.

    R9700 occasionally reports a single dynamic sensor such as socket power as
    ``N/A`` while adjacent one-second samples are complete.  Identity,
    process, ECC semantics, and explicit violations are not retried here.  A
    sensor that remains unavailable for all attempts still fails closed.
    """
    retryable_fragments = (
        "temperature edge is missing or malformed",
        "temperature hotspot is missing or malformed",
        "temperature mem is missing or malformed",
        "GFX clock is missing or malformed",
        "memory clock is missing or malformed",
        "socket power is missing or malformed",
        "metric VRAM used is missing or malformed",
        "metric VRAM total is missing or malformed",
    )
    for attempt in range(METRIC_TELEMETRY_ATTEMPTS):
        try:
            return _metric_evidence_once(target, amd_smi_gpu_index)
        except ContractError as exc:
            if not any(fragment in str(exc) for fragment in retryable_fragments) or attempt + 1 == METRIC_TELEMETRY_ATTEMPTS:
                raise
            time.sleep(METRIC_TELEMETRY_RETRY_SECONDS)
    raise AssertionError("unreachable metric telemetry retry state")


def _vram_auxiliary(target: str, amd_smi_gpu_index: int) -> dict[str, Any]:
    expected = expected_device(target)
    try:
        completed = subprocess.run(
            [AMD_SMI_EXECUTABLE, "monitor", "-v", "-g", expected["gpu_bdf"], "-w", "1", "-i", "1", "--json"],
            capture_output=True, check=False, timeout=MONITOR_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"AMD-SMI VRAM auxiliary sampling failed: {exc}")
    if completed.returncode != 0 or completed.stderr:
        _fail("AMD-SMI VRAM auxiliary sampling did not exit cleanly")
    output = completed.stdout.decode("utf-8", errors="strict")
    json_start = output.find("[")
    if json_start < 0:
        _fail("AMD-SMI VRAM auxiliary sampling has no JSON payload")
    document = parse_json_bytes(output[json_start:].encode("utf-8"), "AMD-SMI VRAM auxiliary sampling")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict) or document[0].get("gpu") != amd_smi_gpu_index:
        _fail("AMD-SMI VRAM auxiliary sampling is not bound to one device")
    data = document[0]
    percent = _number(data.get("vram_percent"), "%", "auxiliary VRAM percent")
    if percent > 100:
        _fail("AMD-SMI auxiliary VRAM percent is outside 0..100")
    return {
        "source": "amd-smi monitor -v", "gpu": amd_smi_gpu_index,
        "used_mb": _number(data.get("vram_used"), "MB", "auxiliary VRAM used"),
        "free_mb": _number(data.get("vram_free"), "MB", "auxiliary VRAM free"),
        "total_mb": _number(data.get("vram_total"), "MB", "auxiliary VRAM total"),
        "percent": percent,
    }


def _amd_smi_observation(target: str, phase: str) -> dict[str, Any]:
    """Observe exactly the matrix device; no ordinal discovery fallback."""
    match, amd_smi_gpu_index = _amd_smi_list_identity(target)
    _static_evidence(target, match, amd_smi_gpu_index)
    metric, _violation = _metric_evidence(target, amd_smi_gpu_index)
    process_doc = _run_json_command([AMD_SMI_EXECUTABLE, "process", "--json", "-g", expected_device(target)["gpu_bdf"]])
    process_list = _process_records(process_doc, amd_smi_gpu_index)
    gpu_processes = _allowed_process_observation(process_list, _parse_phase5_allowed_target_pids())
    children = _child_process_ids(os.getpid())
    ras = metric["ecc_uncorrectable"]
    return {
        "selected_device": expected_device(target),
        "health": {"available": True, "reliable": True, "state": "OK" if ras == 0 else "ERROR", "ras_uncorrectable_count": ras},
        "process": {"available": True, "reliable": True, "state": "CLEAN" if not children else "DIRTY", "gpu_processes": gpu_processes, "residual_runner_children": children},
    }


def _amd_smi_phase_evidence(target: str, phase: str) -> dict[str, Any]:
    match, amd_smi_gpu_index = _amd_smi_list_identity(target)
    static = _static_evidence(target, match, amd_smi_gpu_index)
    metric, violation = _metric_evidence(target, amd_smi_gpu_index)
    vram = _vram_auxiliary(target, amd_smi_gpu_index)
    process_doc = _run_json_command([AMD_SMI_EXECUTABLE, "process", "--json", "-g", expected_device(target)["gpu_bdf"]])
    process_list = _process_records(process_doc, amd_smi_gpu_index)
    _allowed_process_observation(process_list, _parse_phase5_allowed_target_pids())
    return {
        "static": static, "metric": metric, "vram_auxiliary": vram,
        "process_state": "CLEAN", "violation": violation,
    }


def _amd_smi_monitor_sample(target: str, process_pid: int, process_group: int) -> dict[str, Any]:
    _match, amd_smi_gpu_index = _amd_smi_list_identity(target)
    metric, violation = _metric_evidence(target, amd_smi_gpu_index)
    vram = _vram_auxiliary(target, amd_smi_gpu_index)
    process_doc = _run_json_command([AMD_SMI_EXECUTABLE, "process", "--json", "-g", expected_device(target)["gpu_bdf"]])
    process_list = _process_records(process_doc, amd_smi_gpu_index)
    owned_processes = _partition_during_processes(process_list, _parse_phase5_allowed_target_pids())
    members = _process_group_members(process_group)
    pids: set[int] = set()
    for record in owned_processes:
        pids.update(_extract_process_pids(record))
    if not pids:
        raise MonitorNotReady("owned benchmark process is waiting for GPU context registration")
    if process_pid not in members:
        raise MonitorNotReady("owned benchmark process is not visible in its process group yet")
    if not pids.issubset(members):
        _fail("during-run GPU process observation contains a foreign PID")
    loader = _loader_evidence(members)
    if violation["explicit_violation"]:
        _fail("AMD-SMI exposed an explicit throttle or violation during the run")
    return {
        "metric": metric, "vram_auxiliary": vram,
        "process": {"state": "OWNED", "pids": sorted(pids)},
        "loader_path_digest": loader["path_digest"], "loader": loader,
        "violation": violation,
    }


def _monitor_loop(target: str, process_pid: int, monitor_provider: Callable[[str, int, int], dict[str, Any]], stop: threading.Event) -> dict[str, Any]:
    """Collect owned samples with an explicit startup acquisition state.

    GPU process registration and ROCm loader mappings can lag process start, so
    acquisition polls for at most MONITOR_ACQUISITION_TIMEOUT_SECONDS.  Only
    MonitorNotReady is retryable before acquisition; foreign PIDs, invalid
    loader provenance, throttle, and provider errors are hard monitor failures.  Once the
    first sample is acquired, samples are scheduled at one-second cadence and
    process exit ends collection without fabricating a post-exit sample.
    """
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    loader: dict[str, Any] | None = None
    loaders: dict[str, dict[str, Any]] = {}
    acquired = False
    acquisition_deadline = time.monotonic() + MONITOR_ACQUISITION_TIMEOUT_SECONDS
    next_sample = time.monotonic()
    while not stop.is_set():
        if _proc_relationship(process_pid) is None:
            if not acquired:
                errors.append("benchmark process exited before owned GPU/ROCm acquisition")
            break
        try:
            sample = monitor_provider(target, process_pid, process_pid)
            if not isinstance(sample, dict):
                raise ContractError("monitor provider returned a non-object")
            sample = dict(sample)
            sample_loader = sample.pop("loader", None)
            if _proc_relationship(process_pid) is None:
                if not acquired:
                    errors.append("benchmark process exited before the first owned evidence sample")
                break
            if isinstance(sample_loader, dict):
                loader = sample_loader
                digest = loader.get("path_digest")
                if isinstance(digest, str):
                    loaders[digest] = loader
            sample["timestamp_ns"] = time.monotonic_ns()
            samples.append(sample)
            if len(samples) > MAX_MONITOR_SAMPLES:
                errors.append("monitor sample bound exceeded")
                break
            acquired = True
            # Evidence timestamps are recorded after the provider finishes, so
            # schedule from that completion point.  Advancing a prior deadline
            # can make adjacent completion timestamps appear faster than the
            # one-second evidence contract when command latency varies.
            next_sample = time.monotonic() + MONITOR_CADENCE_SECONDS
        except MonitorNotReady as exc:
            if acquired:
                if _proc_relationship(process_pid) is None:
                    break
                # The benchmark can unload ROCm immediately before the runner
                # observes exit and signals this monitor.  Give that teardown
                # signal one acquisition-poll interval; a live process that
                # remains without its acquired loader mappings is still a hard
                # drift failure.
                if stop.wait(MONITOR_ACQUISITION_POLL_SECONDS):
                    break
                if _proc_relationship(process_pid) is None:
                    break
                errors.append(str(exc))
                break
            if time.monotonic() >= acquisition_deadline:
                errors.append(f"bounded GPU/ROCm acquisition timed out: {exc}")
                break
            next_sample = time.monotonic() + MONITOR_ACQUISITION_POLL_SECONDS
            if stop.wait(MONITOR_ACQUISITION_POLL_SECONDS):
                break
            continue
        except (ContractError, OSError, ValueError) as exc:
            errors.append(str(exc))
            break
        delay = max(0.0, next_sample - time.monotonic())
        if stop.wait(delay):
            break
    if not samples and not errors and not stop.is_set():
        errors.append("monitor collected zero successful samples")
    return {
        "samples": samples,
        "errors": errors,
        "loader": loader,
        "loaders": [loaders[digest] for digest in sorted(loaders)],
        "acquisition": "acquired" if acquired else "not-acquired",
    }


def validate_observation(observation: Any, target: str, phase: str) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != {"selected_device", "health", "process"}:
        _fail(f"{phase} health observation is incomplete")
    if observation["selected_device"] != expected_device(target):
        _fail(f"{phase} health selected-device identity is wrong")
    health = observation["health"]
    if not isinstance(health, dict) or health.get("available") is not True or health.get("reliable") is not True or health.get("state") != "OK" or health.get("ras_uncorrectable_count") != 0:
        _fail(f"{phase} health is unavailable, unreliable, unhealthy, or has RAS errors")
    process = observation["process"]
    if not isinstance(process, dict) or process.get("available") is not True or process.get("reliable") is not True or process.get("state") != "CLEAN" or process.get("residual_runner_children") != []:
        _fail(f"{phase} process state is not clean")
    gpu_processes = process.get("gpu_processes")
    allowed_pids = _parse_phase5_allowed_target_pids()
    if not isinstance(gpu_processes, list):
        _fail(f"{phase} GPU process audit is malformed")
    if allowed_pids:
        if not gpu_processes or not isinstance(gpu_processes[0], dict) or set(gpu_processes[0]) != {"allowlisted_pids"}:
            _fail(f"{phase} GPU process allowlist audit is malformed")
        if gpu_processes[0]["allowlisted_pids"] != list(allowed_pids):
            _fail(f"{phase} GPU process allowlist audit is stale")
        records: list[Any] = []
        for entry in gpu_processes[1:]:
            if not isinstance(entry, dict) or set(entry) != {"record", "record_sha256"}:
                _fail(f"{phase} GPU process record audit is malformed")
            records.append(entry["record"])
        if _allowed_process_observation(records, allowed_pids) != gpu_processes:
            _fail(f"{phase} GPU process allowlist audit is not canonical")
    elif gpu_processes != []:
        _fail(f"{phase} process state is not zero-process strict")
    return observation


def _stable_observation_authorization(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validated observation onto fields that authorize a clean run.

    Full AMD-SMI records remain in the observation for audit.  Cumulative and
    diagnostic process counters are deliberately excluded here because they do
    not authorize the allowlisted inert context.
    """
    process = observation["process"]
    gpu_processes = process["gpu_processes"]
    allowed_pids: list[int] = []
    if gpu_processes:
        allowed_pids = list(gpu_processes[0]["allowlisted_pids"])
    # Each present allowlisted record has already passed the strict inert
    # bounds in validate_observation.  Presence is not itself authorization:
    # an external long-lived process may release or lazily recreate an inert
    # context between pre and post without touching the benchmark GPU.
    return {
        "selected_device": observation["selected_device"],
        "health": observation["health"],
        "process": {key: value for key, value in process.items() if key != "gpu_processes"},
        "allowlisted_pids": allowed_pids,
    }


def _observations_have_stable_authorization(
    pre: Mapping[str, Any], post: Mapping[str, Any],
) -> bool:
    return _stable_observation_authorization(pre) == _stable_observation_authorization(post)


def _range(values: list[float]) -> dict[str, float]:
    if not values:
        _fail("monitor summary has no numeric samples")
    return {"min": min(values), "max": max(values)}


def _stable_static_identity(static: Mapping[str, Any]) -> dict[str, Any]:
    """Project static evidence onto fields that must not change during a run.

    AMD-SMI's ``static --clock`` payload contains current levels and, on the
    R9700, even reported frequency-level values that vary between reads.  The
    full payload remains in the evidence, but it is not an identity field.
    Profile and limits have their own explicit drift checks.
    """
    keys = (
        "target", "product", "gpu_bdf", "gpu_uuid", "physical_hip_index",
        "amd_smi_gpu_index", "driver_version", "kernel_version", "vram_total_mb",
    )
    return {key: static.get(key) for key in keys}


def _limit_value(limits: Mapping[str, Any], key: str, unit: str) -> float:
    values = limits.get("values")
    if not isinstance(values, dict):
        _fail("AMD-SMI static limits are missing")
    if key in {"socket_power_limit", "max_power_limit", "min_power_limit"}:
        record = values.get("ppt0", {}).get(key) if isinstance(values.get("ppt0"), dict) else None
    else:
        record = values.get(key)
    return _number(record, unit, f"limit {key}")


def _validate_metric_safety(static: Mapping[str, Any], metrics: list[Mapping[str, Any]], label: str) -> None:
    limits = static.get("limits")
    if not isinstance(limits, dict):
        _fail(f"{label} static limit evidence is missing")
    thresholds = {
        "edge": _limit_value(limits, "slowdown_edge_temperature", "C"),
        "hotspot": _limit_value(limits, "slowdown_hotspot_temperature", "C"),
        "mem": _limit_value(limits, "slowdown_vram_temperature", "C"),
    }
    configured_power_limit = _limit_value(limits, "socket_power_limit", "W")
    maximum_power_limit = _limit_value(limits, "max_power_limit", "W")
    if configured_power_limit > maximum_power_limit:
        _fail(f"{label} configured socket power limit exceeds the published maximum")
    for metric in metrics:
        temperatures = metric.get("temperature_c")
        if not isinstance(temperatures, dict):
            _fail(f"{label} temperature evidence is missing")
        for sensor, threshold in thresholds.items():
            value = temperatures.get(sensor)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                _fail(f"{label} {sensor} temperature evidence is malformed")
            if float(value) >= threshold:
                _fail(f"{label} {sensor} temperature reached the published slowdown limit")
        power = metric.get("power_w")
        if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(float(power)):
            _fail(f"{label} socket power evidence is malformed")


def _validate_loader(loader: Any) -> dict[str, Any]:
    if not isinstance(loader, dict) or set(loader) != {"required_rocm_release", "expected_root", "resolved_paths", "path_digest", "library_digests", "process_ids"}:
        _fail("loader evidence is incomplete")
    if loader["required_rocm_release"] != ROCM_RELEASE or loader["expected_root"] != ROCM_ROOT:
        _fail("loader evidence has the wrong ROCm release/root")
    paths = loader["resolved_paths"]
    if not isinstance(paths, list) or len(paths) < 2 or paths != sorted(set(paths)):
        _fail("loader evidence paths are not canonical")
    if loader["path_digest"] != "sha256:" + hashlib.sha256(canonical_bytes(paths)).hexdigest():
        _fail("loader path digest is stale")
    if not isinstance(loader["library_digests"], dict) or set(loader["library_digests"]) != set(paths):
        _fail("loader library digest set does not match resolved paths")
    if not isinstance(loader["process_ids"], list) or not loader["process_ids"]:
        _fail("loader evidence has no process owner")
    if not any(Path(path).name.startswith("libamdhip64.so") for path in paths) or not any(Path(path).name.startswith("libhsa-runtime64.so") for path in paths):
        _fail("loader evidence does not contain both HIP and ROCr runtime libraries")
    for path_value in paths:
        if not path_value.startswith(ROCM_ROOT + "/"):
            _fail("loader evidence contains an unexpected ROCm root")
        if not re.fullmatch(r"[0-9a-f]{64}", loader["library_digests"][path_value]):
            _fail("loader library digest is malformed")
    return loader


def _validate_phase_evidence(phase: Any, target: str, phase_name: str) -> dict[str, Any]:
    if not isinstance(phase, dict):
        _fail(f"{phase_name} evidence is missing")
    static = phase.get("static")
    metric = phase.get("metric")
    vram = phase.get("vram_auxiliary")
    if not isinstance(static, dict) or static != {**static}:
        _fail(f"{phase_name} static evidence is malformed")
    if static.get("target") != target or static.get("gpu_bdf") != expected_device(target)["gpu_bdf"] or static.get("gpu_uuid") != expected_device(target)["gpu_uuid"] or static.get("product") != expected_device(target)["product"]:
        _fail(f"{phase_name} static exact identity is stale")
    if not isinstance(metric, dict) or not isinstance(vram, dict):
        _fail(f"{phase_name} metric/VRAM evidence is missing")
    if metric.get("ecc_uncorrectable") != 0 or metric.get("throttle_status") not in {"UNTHROTTLED", "THROTTLED"}:
        _fail(f"{phase_name} health contains ECC or malformed throttle evidence")
    if phase.get("process_state") != "CLEAN":
        _fail(f"{phase_name} process state is not clean")
    if vram.get("source") != "amd-smi monitor -v":
        _fail(f"{phase_name} is missing AMD-SMI auxiliary VRAM evidence")
    return phase


def _validate_monitor_capture(capture: Mapping[str, Any], target: str, expected_pid: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    monitor = capture.get("monitor")
    if not isinstance(monitor, dict) or not isinstance(monitor.get("samples"), list) or not isinstance(monitor.get("errors"), list):
        _fail("during-run monitor evidence is missing or malformed")
    if monitor["errors"]:
        _fail("during-run monitor evidence contains errors: " + "; ".join(str(item) for item in monitor["errors"]))
    samples = monitor["samples"]
    if not samples:
        _fail("during-run monitor collected zero samples")
    loader_records = monitor.get("loaders")
    if not isinstance(loader_records, list) or not loader_records:
        _fail("during-run monitor has no loader provenance records")
    validated_loaders: dict[str, dict[str, Any]] = {}
    for loader_record in loader_records:
        validated = _validate_loader(loader_record)
        digest = validated["path_digest"]
        if digest in validated_loaders and validated_loaders[digest] != validated:
            _fail("during-run loader provenance digest is ambiguous")
        validated_loaders[digest] = validated
    normalized: list[dict[str, Any]] = []
    previous_timestamp: int | None = None
    for sample in samples:
        if not isinstance(sample, dict):
            _fail("during-run monitor sample is not an object")
        metric = sample.get("metric")
        vram = sample.get("vram_auxiliary")
        process = sample.get("process")
        loader_path_digest = sample.get("loader_path_digest")
        if not isinstance(metric, dict) or not isinstance(vram, dict) or not isinstance(process, dict) or not isinstance(loader_path_digest, str):
            _fail("during-run monitor sample is incomplete")
        if metric.get("ecc_uncorrectable") != 0 or metric.get("throttle_status") not in {"UNTHROTTLED", "THROTTLED"}:
            _fail("during-run monitor found ECC or malformed throttle status")
        if vram.get("source") != "amd-smi monitor -v" or process.get("state") != "OWNED" or not process.get("pids"):
            _fail("during-run monitor lacks owned process or auxiliary VRAM evidence")
        if expected_pid is not None and expected_pid not in process["pids"]:
            _fail("during-run monitor process ownership does not name the benchmark PID")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", loader_path_digest):
            _fail("during-run loader path digest is malformed")
        if loader_path_digest not in validated_loaders:
            _fail("during-run loader evidence digest has no validated provenance record")
        timestamp = sample.get("timestamp_ns")
        if not isinstance(timestamp, int) or timestamp < 0:
            _fail("during-run monitor timestamp is malformed")
        if previous_timestamp is not None and timestamp - previous_timestamp < int(MONITOR_CADENCE_SECONDS * 1_000_000_000):
            _fail("during-run monitor cadence is faster than the one-second contract")
        previous_timestamp = timestamp
        normalized.append({
            "timestamp_ns": timestamp, "metric": metric,
            "vram_auxiliary": vram, "process": process,
            "loader_path_digest": loader_path_digest, "violation": sample.get("violation", {}),
        })
    # ROCm components may be mapped lazily after the first GPU operation.  Each
    # sample above is independently constrained to the pinned ROCm root and
    # content digests; requiring the path set itself to remain identical would
    # reject legitimate lazy additions.
    final_loader_digest = normalized[-1]["loader_path_digest"]
    metrics = [sample["metric"] for sample in normalized]
    vram_metrics = [sample["vram_auxiliary"] for sample in normalized]
    power_statuses = [metric["throttle_status"] for metric in metrics]
    perf_levels = sorted({metric["perf_level"] for metric in metrics})
    summary = {
        "sample_count": len(normalized),
        "temperature_hotspot_c": _range([float(metric["temperature_c"]["hotspot"]) for metric in metrics]),
        "temperature_mem_c": _range([float(metric["temperature_c"]["mem"]) for metric in metrics]),
        "gfx_clock_mhz": _range([float(metric["gfx_clock_mhz"]) for metric in metrics]),
        "mem_clock_mhz": _range([float(metric["mem_clock_mhz"]) for metric in metrics]),
        "power_w": _range([float(metric["power_w"]) for metric in metrics]),
        "vram_used_mb": _range([float(metric["vram_used_mb"]) for metric in metrics]),
        "vram_aux_used_mb": _range([float(metric["used_mb"]) for metric in vram_metrics]),
        "perf_levels": perf_levels,
    }
    violation = {
        "power_statuses": sorted(set(power_statuses)),
        "explicit_violation": False,
        "accumulator_available": False,
        "accumulator_reason": "AMD-SMI monitor samples require the phase metric evidence for accumulators",
        "accumulator_digest": "sha256:" + hashlib.sha256(canonical_bytes([sample.get("violation", {}) for sample in normalized])).hexdigest(),
    }
    for sample in normalized:
        sample_violation = sample.get("violation")
        if isinstance(sample_violation, dict):
            violation["explicit_violation"] = violation["explicit_violation"] or bool(sample_violation.get("explicit_violation"))
            violation["accumulator_available"] = violation["accumulator_available"] or bool(sample_violation.get("accumulator_available"))
            if isinstance(sample_violation.get("accumulator_reason"), str):
                violation["accumulator_reason"] = sample_violation["accumulator_reason"]
    if violation["explicit_violation"]:
        _fail("during-run monitor found an explicit throttle/violation")
    if not isinstance(capture.get("pid"), int) and expected_pid is None:
        expected_pid = None
    return normalized, {"summary": summary, "violation": violation, "loader_path_digest": final_loader_digest}


def _build_evidence(pre: Mapping[str, Any], post: Mapping[str, Any], capture: Mapping[str, Any], target: str, tool: Mapping[str, str]) -> dict[str, Any]:
    pre_phase = _validate_phase_evidence(pre, target, "pre")
    post_phase = _validate_phase_evidence(post, target, "post")
    _validate_metric_safety(pre_phase["static"], [pre_phase["metric"]], "pre")
    _validate_metric_safety(post_phase["static"], [post_phase["metric"]], "post")
    expected_pid = capture.get("pid") if isinstance(capture.get("pid"), int) else None
    samples, during_info = _validate_monitor_capture(capture, target, expected_pid)
    _validate_metric_safety(pre_phase["static"], [sample["metric"] for sample in samples], "during-run")
    loader = capture.get("loader")
    if loader is None and isinstance(capture.get("monitor"), dict):
        loader = capture["monitor"].get("loader")
    if loader is None and isinstance(capture.get("monitor"), dict) and capture["monitor"].get("samples"):
        loader = capture["monitor"]["samples"][-1].get("loader")
    loader = _validate_loader(loader)
    if loader["path_digest"] != during_info["loader_path_digest"]:
        _fail("loader path digest does not match during-run samples")
    sample_digest = "sha256:" + hashlib.sha256(canonical_bytes(samples)).hexdigest()
    process_sample_digest = "sha256:" + hashlib.sha256(canonical_bytes([sample["process"] for sample in samples])).hexdigest()
    during = {
        "sample_count": len(samples), "sample_digest": sample_digest,
        "first": samples[0], "last": samples[-1], "summary": during_info["summary"],
        "process_sample_digest": process_sample_digest, "loader": loader, "violation": during_info["violation"],
    }
    explicit = bool(pre.get("violation", {}).get("explicit_violation") or post.get("violation", {}).get("explicit_violation") or during_info["violation"]["explicit_violation"])
    if explicit:
        _fail("AMD-SMI evidence contains an explicit throttle/violation")
    checks = {
        "exact_identity": _stable_static_identity(pre_phase["static"]) == _stable_static_identity(post_phase["static"]),
        "static_identity_unchanged": pre_phase["static"]["target"] == post_phase["static"]["target"] and pre_phase["static"]["product"] == post_phase["static"]["product"],
        "profile_unchanged": pre_phase["static"]["profile"] == post_phase["static"]["profile"],
        "limits_unchanged": pre_phase["static"]["limits"] == post_phase["static"]["limits"],
        "performance_level_unchanged": pre_phase["metric"]["perf_level"] == post_phase["metric"]["perf_level"] and len(during_info["summary"]["perf_levels"]) == 1 and during_info["summary"]["perf_levels"][0] == pre_phase["metric"]["perf_level"],
        "explicit_violation": False,
        "vram_auxiliary_complete": True,
        "process_ownership": all(sample["process"]["state"] == "OWNED" for sample in samples),
        "loader_paths_verified": True,
        "monitor_errors": 0,
        "process_group_cleanup": capture.get("process_group_gone") is True,
    }
    if not all(value is True or value == 0 for value in checks.values()):
        _fail("during-run evidence checks did not all pass")
    return {
        "version": EVIDENCE_VERSION, "cadence_seconds": 1,
        "tool": dict(tool),
        "definitions": {
            "clock_variation": "Dynamic clock min/max is observational; no numeric threshold is a violation.",
            "violation": "When violation accumulators are unavailable, aggregate THROTTLED status is observational; ECC, published thermal/power limits, and exposed active violations remain fail-closed.",
            "process_ownership": "Every during sample must name only descendants of the benchmark process group.",
        },
        "visibility": {"cleared": list(VISIBILITY_NAMES), "selector": "ROCR_VISIBLE_DEVICES", "uuid": expected_device(target)["gpu_uuid"]},
        "pre": {key: pre_phase[key] for key in ("static", "metric", "vram_auxiliary", "process_state")},
        "during": during,
        "post": {key: post_phase[key] for key in ("static", "metric", "vram_auxiliary", "process_state")},
        "checks": checks,
    }


def _failed_evidence(pre: Mapping[str, Any], post: Mapping[str, Any], capture: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Keep a schema-valid, explicitly failed record when runtime evidence is bad."""
    pre_phase = {key: pre[key] for key in ("static", "metric", "vram_auxiliary", "process_state")}
    post_phase = {key: post[key] for key in ("static", "metric", "vram_auxiliary", "process_state")}
    target = pre_phase["static"]["target"]
    paths = [ROCM_ROOT + "/lib/libamdhip64.so.7.14.60850-0000000", ROCM_ROOT + "/lib/libhsa-runtime64.so.1.21.0"]
    path_digest = "sha256:" + hashlib.sha256(canonical_bytes(paths)).hexdigest()
    metric = pre_phase["metric"]
    vram = pre_phase["vram_auxiliary"]
    sample = {
        "timestamp_ns": 0, "metric": metric, "vram_auxiliary": vram,
        "process": {"state": "OWNED", "pids": [capture.get("pid", 1) if isinstance(capture.get("pid", 1), int) else 1]},
        "loader_path_digest": path_digest,
        "violation": {"power_statuses": ["UNAVAILABLE"], "explicit_violation": True, "accumulator_available": False, "accumulator_reason": reason, "accumulator_digest": "sha256:" + "0" * 64},
    }
    loader = {"required_rocm_release": ROCM_RELEASE, "expected_root": ROCM_ROOT, "resolved_paths": paths, "path_digest": path_digest, "library_digests": {path: "0" * 64 for path in paths}, "process_ids": sample["process"]["pids"]}
    zero_range = {"min": 0, "max": 0}
    summary = {"sample_count": 1, "temperature_hotspot_c": zero_range, "temperature_mem_c": zero_range, "gfx_clock_mhz": zero_range, "mem_clock_mhz": zero_range, "power_w": zero_range, "vram_used_mb": zero_range, "vram_aux_used_mb": zero_range, "perf_levels": ["unavailable"]}
    return {
        "version": EVIDENCE_VERSION, "cadence_seconds": 1,
        "tool": {"path": AMD_SMI_EXECUTABLE, "tool_version": "unavailable", "library_version": "unavailable", "rocm_version": ROCM_RELEASE},
        "definitions": {"clock_variation": "Dynamic clock min/max is observational; no numeric threshold is a violation.", "violation": "When violation accumulators are unavailable, aggregate THROTTLED status is observational; ECC, published thermal/power limits, and exposed active violations remain fail-closed.", "process_ownership": "Every during sample must name only descendants of the benchmark process group."},
        "visibility": {"cleared": list(VISIBILITY_NAMES), "selector": "ROCR_VISIBLE_DEVICES", "uuid": expected_device(target)["gpu_uuid"]},
        "pre": pre_phase, "during": {"sample_count": 1, "sample_digest": "sha256:" + hashlib.sha256(canonical_bytes([sample])).hexdigest(), "first": sample, "last": sample, "summary": summary, "process_sample_digest": "sha256:" + hashlib.sha256(canonical_bytes([sample["process"]])).hexdigest(), "loader": loader, "violation": {"power_statuses": ["UNAVAILABLE"], "explicit_violation": True, "accumulator_available": False, "accumulator_reason": reason, "accumulator_digest": "sha256:" + "0" * 64}}, "post": post_phase,
        "checks": {"exact_identity": False, "static_identity_unchanged": False, "profile_unchanged": False, "limits_unchanged": False, "performance_level_unchanged": False, "explicit_violation": True, "vram_auxiliary_complete": False, "process_ownership": False, "loader_paths_verified": False, "monitor_errors": 1, "process_group_cleanup": False},
    }


def _fallback_observation(target: str) -> dict[str, Any]:
    return {
        "selected_device": expected_device(target),
        "health": {"available": False, "reliable": False, "state": "UNAVAILABLE", "ras_uncorrectable_count": None},
        "process": {"available": False, "reliable": False, "state": "UNAVAILABLE", "gpu_processes": [], "residual_runner_children": []},
    }


def _fallback_phase_evidence(target: str) -> dict[str, Any]:
    device = expected_device(target)
    static = {
        "target": target, "product": device["product"], "gpu_bdf": device["gpu_bdf"], "gpu_uuid": device["gpu_uuid"],
        "physical_hip_index": device["physical_hip_index"], "amd_smi_gpu_index": device["physical_hip_index"],
        "driver_version": "unavailable", "kernel_version": "unavailable",
        "profile": {"current": "unavailable", "available_profiles": ["unavailable"], "digest": "sha256:" + "0" * 64},
        "limits": {"values": {}, "digest": "sha256:" + "0" * 64},
        "clock_levels": {"values": {}, "digest": "sha256:" + "0" * 64}, "vram_total_mb": 1,
    }
    metric = {
        "temperature_c": {"edge": 0, "hotspot": 0, "mem": 0}, "gfx_clock_mhz": 0, "mem_clock_mhz": 0,
        "power_w": 0, "perf_level": "unavailable", "throttle_status": "UNTHROTTLED", "vram_used_mb": 0,
        "vram_total_mb": 1, "ecc_uncorrectable": 0, "metric_digest": "sha256:" + "0" * 64,
    }
    vram = {"source": "amd-smi monitor -v", "gpu": device["physical_hip_index"], "used_mb": 0, "free_mb": 1, "total_mb": 1, "percent": 0}
    return {"static": static, "metric": metric, "vram_auxiliary": vram, "process_state": "CLEAN", "violation": {"power_statuses": ["UNAVAILABLE"], "explicit_violation": True, "accumulator_available": False, "accumulator_reason": "failure manifest placeholder", "accumulator_digest": "sha256:" + "0" * 64}}


def _durable_failure_manifest(
    row: Mapping[str, Any], matrix_path: Path, matrix_digest: str, output_dir: Path,
    reason: str, *, binary: Path, build_manifest: Path | None, build_document: Mapping[str, Any] | None = None,
    capture: Mapping[str, Any] | None = None, pre: Mapping[str, Any] | None = None,
    post: Mapping[str, Any] | None = None, pre_evidence: Mapping[str, Any] | None = None,
    post_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an explicit FAIL record even when setup or monitoring aborts."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw-result.json"
    manifest_path = output_dir / "report.json"
    if raw_path.exists() or manifest_path.exists() or raw_path.is_symlink() or manifest_path.is_symlink():
        _fail(f"benchmark output directory already contains an evidence file: {output_dir}")
    raw_bytes = canonical_bytes({"benchmark_schema_version": DIRECT_VERSION, "state": "FAIL", "failure_reason": reason})
    _write(raw_path, raw_bytes, "failure raw result")
    try:
        binary_path = binary.resolve()
        binary_sha = sha256_file(binary_path, "benchmark binary", max_bytes=MAX_BINARY_BYTES) if binary_path.is_file() and not binary_path.is_symlink() else "0" * 64
        binary_bytes = max(1, binary_path.stat().st_size) if binary_path.is_file() and not binary_path.is_symlink() else 1
    except OSError:
        binary_path, binary_sha, binary_bytes = binary, "0" * 64, 1
    document = dict(build_document or {})
    build = {
        "path": str(build_manifest.resolve()) if build_manifest is not None else str(output_dir / "build-identity.json"),
        "sha256": sha256_file(build_manifest.resolve(), "build identity manifest", max_bytes=4 * 1024 * 1024) if build_manifest is not None and build_manifest.is_file() and not build_manifest.is_symlink() else "0" * 64,
        "source_root": document.get("source_root", str(MATRIX_PATH.parents[2].resolve())),
        "source_base_revision": document.get("source_base_revision", "0" * 40),
        "semantic_tree": document.get("semantic_tree", "0" * 40),
        "build_inputs_digest": document.get("build_inputs_digest", "sha256:" + "0" * 64),
        "build_configuration": expected_build_configuration(row["target"]),
        "target": row["target"], "backend": "hip", "rocm_release": ROCM_RELEASE, "rocm_root": ROCM_ROOT,
        "binary_sha256": binary_sha,
    }
    fallback_pre = dict(pre or _fallback_observation(row["target"]))
    fallback_post = dict(post or _fallback_observation(row["target"]))
    fallback_pre_evidence = dict(pre_evidence or _fallback_phase_evidence(row["target"]))
    fallback_post_evidence = dict(post_evidence or _fallback_phase_evidence(row["target"]))
    evidence = _failed_evidence(fallback_pre_evidence, fallback_post_evidence, capture or {}, reason)
    capture = capture or {}
    execution = {
        "exit_code": capture.get("exit_code"), "timed_out": bool(capture.get("timed_out", False)),
        "timeout_seconds": row["timeout_seconds"], "stderr_bytes": len(capture.get("stderr", b"")),
        "term_sent": bool(capture.get("term_sent", False)), "kill_sent": bool(capture.get("kill_sent", False)),
        "process_group_gone": capture.get("process_group_gone") is True,
    }
    cleanup = {
        "pre_process_clean": fallback_pre.get("process", {}).get("state") == "CLEAN",
        "post_process_clean": fallback_post.get("process", {}).get("state") == "CLEAN",
        "process_group_gone": capture.get("process_group_gone") is True,
        "retryable_cleanup": 0 if capture.get("process_group_gone") is True else 1,
        "durable_quarantine": 0,
    }
    manifest = {
        "benchmark_schema_version": DIRECT_VERSION, "record_kind": "evidence_manifest", "state": "FAIL", "required": False,
        "failure_reason": reason, "row_id": row["row_id"], "claims": dict(CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": DIRECT_VERSION, "sha256": matrix_digest},
        "binary": {"path": str(binary_path), "sha256": binary_sha, "bytes": binary_bytes}, "build_identity": build,
        "model_lock": {"path": str((MATRIX_PATH.parents[2] / expected_model(row["model_size"])["lock_path"]).resolve()), "sha256": "0" * 64, "fingerprint": expected_model(row["model_size"])["lock_fingerprint"]},
        "model_cache": {"path": str(output_dir / "model-cache"), "sha256": "0" * 64},
        "raw_artifact": {"path": str(raw_path), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "bytes": len(raw_bytes)},
        "observations": {"pre": fallback_pre, "post": fallback_post}, "evidence": evidence,
        "execution": execution, "cleanup": cleanup,
    }
    schema_validate(manifest, DIRECT_SCHEMA_PATH, "failure performance evidence manifest", "manifest")
    encoded = canonical_bytes(manifest)
    _write(manifest_path, encoded, "failure performance evidence manifest")
    _write(manifest_path.with_name("report.json.sha256"), f"{hashlib.sha256(encoded).hexdigest()}  report.json\n".encode("ascii"), "failure manifest digest sidecar")
    _write(raw_path.with_name("raw-result.json.sha256"), f"{hashlib.sha256(raw_bytes).hexdigest()}  raw-result.json\n".encode("ascii"), "failure raw result digest sidecar")
    return manifest


def run_row(
    row_id: str,
    binary: Path,
    model_lock: Path,
    model_cache: Path,
    output_dir: Path,
    *,
    build_manifest: Path | None = None,
    matrix_path: Path = MATRIX_PATH,
    repo: Path | None = None,
    command_runner: Callable[[list[str], Mapping[str, str], Path, int], dict[str, Any]] | None = None,
    observation_provider: Callable[[str, str], dict[str, Any]] | None = None,
    evidence_provider: Callable[[str, str], dict[str, Any]] | None = None,
    tool_provider: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    repo = (repo or MATRIX_PATH.parents[2]).resolve()
    matrix, matrix_digest = load_matrix(matrix_path)
    rows = {row["row_id"]: row for row in matrix["rows"]}
    if row_id not in rows:
        _fail(f"row is not in the closed performance matrix: {row_id}")
    row = rows[row_id]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw-result.json"
    manifest_path = output_dir / "report.json"
    if raw_path.exists() or manifest_path.exists() or raw_path.is_symlink() or manifest_path.is_symlink():
        _fail("benchmark output directory already contains an evidence file")
    binary_input = binary
    try:
        binary = _regular_executable(binary)
        lock = _resolve_repo_path(str(model_lock), repo, "model lock")
        cache = _resolve_repo_path(str(model_cache), repo, "model cache")
        lock_document, lock_digest = _validate_lock(lock, row["model_size"], repo)
        cache_hash = _validate_cache(cache, lock_document)
        binary_digest = sha256_file(binary, "benchmark binary", max_bytes=MAX_BINARY_BYTES)
        if build_manifest is None:
            _fail("an immutable build identity manifest is required")
        build_document, build_digest = _validate_build_manifest(build_manifest, binary, row["target"], repo)
    except ContractError as exc:
        return _durable_failure_manifest(row, matrix_path, matrix_digest, output_dir, str(exc), binary=binary_input, build_manifest=build_manifest)
    expected = expected_device(row["target"])
    observer = observation_provider or _amd_smi_observation
    evidence_observer = evidence_provider or _amd_smi_phase_evidence
    try:
        pre = validate_observation(observer(row["target"], "pre"), row["target"], "pre")
        pre_evidence = evidence_observer(row["target"], "pre")
    except (ContractError, OSError, ValueError) as exc:
        return _durable_failure_manifest(row, matrix_path, matrix_digest, output_dir, f"pre-run health/evidence failed: {exc}", binary=binary, build_manifest=build_manifest, build_document=build_document)
    command = _expected_command(binary, row, lock, cache)
    environment = _execution_environment(row_id, row["target"])
    capture: dict[str, Any] = {"stdout": b"", "stderr": b"", "exit_code": None, "timed_out": False, "term_sent": False, "kill_sent": False, "process_group_gone": False, "output_overflow": [], "monitor": {"samples": [], "errors": []}}
    post: dict[str, Any] = _fallback_observation(row["target"])
    post_evidence: dict[str, Any] = _fallback_phase_evidence(row["target"])
    reasons: list[str] = []
    try:
        if command_runner:
            capture = command_runner(command, environment, repo, row["timeout_seconds"])
        else:
            capture = _execute_bounded(
                command, environment, repo, row["timeout_seconds"],
                monitor_provider=_amd_smi_monitor_sample, monitor_target=row["target"],
            )
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"benchmark execution failed: {exc}")
    try:
        post = validate_observation(observer(row["target"], "post"), row["target"], "post")
        post_evidence = evidence_observer(row["target"], "post")
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"post-run health/evidence failed: {exc}")
    _write(raw_path, capture.get("stdout", b""), "raw result")
    raw_digest = sha256_file(raw_path, "raw result", max_bytes=MAX_RAW_BYTES)
    raw_bytes = raw_path.stat().st_size
    try:
        raw, _, _ = read_json(raw_path, "raw result", MAX_RAW_BYTES)
        validate_cli_result(raw, row)
    except ContractError as exc:
        reasons.append(str(exc))
    if capture.get("exit_code") != 0:
        reasons.append(f"benchmark process exited with {capture.get('exit_code')}")
    if capture.get("timed_out"):
        reasons.append("benchmark process timed out")
    if capture.get("output_overflow"):
        reasons.append(
            "benchmark output exceeded the bounded limit on "
            + ", ".join(str(name) for name in capture["output_overflow"])
        )
    if capture.get("stderr", b"") != b"":
        reasons.append("benchmark stderr was not empty")
    if capture.get("process_group_gone") is not True:
        reasons.append("benchmark process group cleanup was not clean")
    if not _observations_have_stable_authorization(pre, post):
        reasons.append("pre/post health or process authorization differs")
    try:
        tool = tool_provider() if tool_provider else {"path": AMD_SMI_EXECUTABLE, **_amd_smi_version()}
        evidence = _build_evidence(pre_evidence, post_evidence, capture, row["target"], tool)
    except ContractError as exc:
        evidence = _failed_evidence(pre_evidence, post_evidence, capture, str(exc))
        reasons.append(f"runtime evidence validation failed: {exc}")
    try:
        binary_after = sha256_file(binary, "benchmark binary after run", max_bytes=MAX_BINARY_BYTES)
        if binary_after != binary_digest:
            reasons.append("benchmark binary changed during the run")
    except ContractError as exc:
        reasons.append(f"benchmark binary post-run validation failed: {exc}")
    try:
        build_after = sha256_file(build_manifest, "build identity manifest after run", max_bytes=4 * 1024 * 1024)
        if build_after != build_digest:
            reasons.append("build identity manifest changed during the run")
    except ContractError as exc:
        reasons.append(f"build identity manifest post-run validation failed: {exc}")
    try:
        cache_after = cache_digest(cache)
    except ContractError as exc:
        cache_after = "0" * 64
        reasons.append(str(exc))
    if cache_after != cache_hash:
        reasons.append("model cache changed during the run")
    cleanup = {
        "pre_process_clean": pre["process"]["state"] == "CLEAN",
        "post_process_clean": post["process"]["state"] == "CLEAN",
        "process_group_gone": capture.get("process_group_gone") is True,
        "retryable_cleanup": 0 if capture.get("process_group_gone") is True and post["process"]["state"] == "CLEAN" else 1,
        "durable_quarantine": 0,
    }
    execution = {
        "exit_code": capture.get("exit_code"), "timed_out": capture.get("timed_out"),
        "timeout_seconds": row["timeout_seconds"], "stderr_bytes": len(capture.get("stderr", b"")),
        "term_sent": capture.get("term_sent"), "kill_sent": capture.get("kill_sent"),
        "process_group_gone": capture.get("process_group_gone"),
    }
    manifest = {
        "benchmark_schema_version": DIRECT_VERSION,
        "record_kind": "evidence_manifest",
        "state": "PASS" if not reasons else "FAIL",
        "required": False,
        "failure_reason": "; ".join(reasons) if reasons else None,
        "row_id": row_id,
        "claims": dict(CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": DIRECT_VERSION, "sha256": matrix_digest},
        "binary": {"path": str(binary), "sha256": binary_digest, "bytes": binary.stat().st_size},
        "build_identity": {
            "path": str(build_manifest.resolve()), "sha256": build_digest,
            "source_root": build_document["source_root"], "source_base_revision": build_document["source_base_revision"],
            "semantic_tree": build_document["semantic_tree"],
            "build_inputs_digest": build_document["build_inputs_digest"],
            "build_configuration": build_document["build_configuration"], "target": build_document["target"],
            "backend": build_document["backend"], "rocm_release": build_document["rocm_release"],
            "rocm_root": build_document["rocm_root"], "binary_sha256": build_document["binary_sha256"],
        },
        "model_lock": {"path": str(lock), "sha256": lock_digest, "fingerprint": expected_model(row["model_size"])["lock_fingerprint"]},
        "model_cache": {"path": str(cache), "sha256": cache_hash},
        "raw_artifact": {"path": str(raw_path), "sha256": raw_digest, "bytes": raw_bytes},
        "observations": {"pre": pre, "post": post},
        "evidence": evidence,
        "execution": execution,
        "cleanup": cleanup,
    }
    schema_validate(manifest, DIRECT_SCHEMA_PATH, "performance evidence manifest", "manifest")
    encoded = canonical_bytes(manifest)
    _write(manifest_path, encoded, "performance evidence manifest")
    _write(manifest_path.with_name("report.json.sha256"), f"{hashlib.sha256(encoded).hexdigest()}  report.json\n".encode("ascii"), "manifest digest sidecar")
    _write(raw_path.with_name("raw-result.json.sha256"), f"{raw_digest}  raw-result.json\n".encode("ascii"), "raw result digest sidecar")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--repo", type=Path, default=MATRIX_PATH.parents[2])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_row(args.row, args.binary, args.model_lock, args.model_cache, args.output_dir, build_manifest=args.build_manifest, matrix_path=args.matrix, repo=args.repo)
    except (ContractError, OSError, ValueError) as exc:
        print(f"engine-performance: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"engine-performance: {manifest['state']}")
    return 0 if manifest["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
