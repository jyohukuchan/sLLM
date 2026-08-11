#!/usr/bin/env python3
"""Build exactly the dedicated P0 public RMSNorm producer.

The builder never executes the binary.  It uses a fresh Cargo target
directory, copies only the canonical release binary into the artifact
directory, records the complete source/build identity, and validates the
result before returning.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from common import ContractError, ROOT, canonical_bytes, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_p0_contracts import (  # noqa: E402
    P0_BINARY,
    P0_BINARY_ROLE,
    P0_BUILD_COMMAND,
    P0_BUILD_KILL_GRACE_SECONDS,
    P0_BUILD_LIMITS,
    P0_BUILD_OUTPUT_LIMIT_BYTES,
    P0_BUILD_TIMEOUT_SECONDS,
    P0_SIDECAR,
    PRODUCER_STATUS,
    PUBLIC_PATH,
    DTYPE_CONTRACT,
    p0_build_environment,
    validate_artifact,
    validate_candidate,
    source_set,
)


def _regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} cannot be stated: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    if executable and metadata.st_mode & 0o111 == 0:
        raise ContractError(f"{label} must be executable")


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reviewed_sha": args.reviewed_sha,
        "tested_sha": args.tested_sha,
        "workflow_sha": args.workflow_sha,
        "git_tree_oid": args.tree_oid,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }

def _read_process_identity(pid: int) -> tuple[int, int, int] | None:
    """Return ``(starttime, session, process_group)`` from Linux procfs."""

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot inspect P0 Cargo PID {pid}: {exc}") from exc
    right_paren = stat_text.rfind(")")
    if right_paren < 0:
        raise ContractError(f"malformed /proc stat for P0 Cargo PID {pid}")
    fields = stat_text[right_paren + 2 :].split()
    if len(fields) < 20:
        raise ContractError(f"short /proc stat for P0 Cargo PID {pid}")
    try:
        return int(fields[19]), int(fields[3]), int(fields[2])
    except ValueError as exc:
        raise ContractError(f"invalid /proc stat for P0 Cargo PID {pid}") from exc

ProcessAnchor = tuple[int, int, int, int]

def _capture_process_anchor(process: subprocess.Popen[bytes]) -> ProcessAnchor:
    identity = _read_process_identity(process.pid)
    if identity is None:
        raise ContractError("P0 Cargo leader disappeared before its identity was captured")
    starttime, session, process_group = identity
    if process_group != process.pid or session != process.pid:
        raise ContractError("P0 Cargo leader did not retain its private session and process group")
    return process.pid, starttime, session, process_group

def _validate_process_group_anchor(anchor: ProcessAnchor) -> None:
    current = _read_process_identity(anchor[0])
    if current != anchor[1:]:
        raise ContractError("P0 Cargo process-group anchor identity changed")

def _process_group_members(anchor: ProcessAnchor) -> tuple[int, ...]:
    members: list[int] = []
    try:
        entries = Path("/proc").iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            identity = _read_process_identity(pid)
            if identity is not None and identity[1:] == anchor[2:]:
                members.append(pid)
    except OSError as exc:
        raise ContractError(f"cannot inspect P0 Cargo process group: {exc}") from exc
    return tuple(sorted(members))

def _signal_process_group(group_id: int, signal_value: signal.Signals) -> None:
    try:
        os.killpg(group_id, signal_value)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ContractError(f"P0 Cargo process-group signal failed: {exc}") from exc

def _waitid_without_reap(pid: int) -> Any | None:
    result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is None or result.si_pid == 0:
        return None
    if result.si_pid != pid:
        raise ContractError("P0 Cargo waitid returned an unexpected PID")
    return result

def _exit_code(result: Any) -> int:
    if result.si_code == os.CLD_EXITED:
        return int(result.si_status)
    if result.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        return -int(result.si_status)
    raise ContractError("P0 Cargo returned an unsupported wait status")

def _record_cleanup_error(errors: list[str], phase: str, exc: BaseException) -> None:
    errors.append(f"{phase}: {type(exc).__name__}: {exc}")


def _attempt_cleanup(errors: list[str], phase: str, action: Any) -> None:
    try:
        action()
    except BaseException as exc:
        _record_cleanup_error(errors, phase, exc)


def _confirm_process_group_disappearance(
    errors: list[str], anchor: ProcessAnchor | None
) -> None:
    if anchor is None:
        errors.append("post-KILL process group disappearance not proven: process-group anchor unavailable")
        return
    deadline = time.monotonic() + P0_BUILD_KILL_GRACE_SECONDS
    members: tuple[int, ...] = ()
    while True:
        try:
            members = _process_group_members(anchor)
        except BaseException as exc:
            _record_cleanup_error(errors, "post-KILL group inspection", exc)
            break
        if not members:
            return
        if deadline - time.monotonic() <= 0:
            break
        try:
            time.sleep(min(0.01, deadline - time.monotonic()))
        except BaseException as exc:
            _record_cleanup_error(errors, "post-KILL disappearance sleep", exc)
            break

    if members:
        errors.append(f"post-KILL process group did not disappear; retained members: {','.join(str(pid) for pid in members)}")
        return
    errors.append("post-KILL process group disappearance not proven")


def _cleanup_process_group(
    process: subprocess.Popen[bytes], anchor: ProcessAnchor | None
) -> list[str]:
    errors: list[str] = []
    group_id = anchor[3] if anchor is not None else process.pid
    try:
        _attempt_cleanup(errors, "TERM", lambda: _signal_process_group(group_id, signal.SIGTERM))

        try:
            grace_deadline = time.monotonic() + P0_BUILD_KILL_GRACE_SECONDS
            while time.monotonic() < grace_deadline:
                if _waitid_without_reap(process.pid) is not None:
                    break
                time.sleep(0.01)
        except BaseException as exc:
            _record_cleanup_error(errors, "grace", exc)

        _attempt_cleanup(errors, "KILL", lambda: _signal_process_group(group_id, signal.SIGKILL))

        reaped = False
        try:
            reap_deadline = time.monotonic() + P0_BUILD_KILL_GRACE_SECONDS
            for _ in range(128):
                if time.monotonic() >= reap_deadline:
                    break
                try:
                    waited_pid, status = os.waitpid(process.pid, os.WNOHANG)
                except ChildProcessError:
                    process.returncode = -int(signal.SIGKILL)
                    reaped = True
                    break
                except BaseException as exc:
                    _record_cleanup_error(errors, "leader reap", exc)
                    continue
                if waited_pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    reaped = True
                    break
                try:
                    time.sleep(0.01)
                except BaseException as exc:
                    _record_cleanup_error(errors, "reap sleep", exc)
                    continue
        except BaseException as exc:
            _record_cleanup_error(errors, "leader reap", exc)
        if not reaped:
            errors.append("leader reap: bounded direct reap did not prove completion")
    finally:
        _confirm_process_group_disappearance(errors, anchor)
    return errors

def _close_resource(resource: Any, label: str, errors: list[str]) -> None:
    if resource is None:
        return
    try:
        resource.close()
    except BaseException as exc:
        _record_cleanup_error(errors, f"close {label}", exc)


def _run_bounded_build(
    command: list[str], *, cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[bytes]:
    """Run the fixed Cargo build with deadline, output bound, and group cleanup."""

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ContractError(f"dedicated P0 Cargo build failed to start: {exc}") from exc
    selector: selectors.BaseSelector | None = None
    anchor: ProcessAnchor | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    result: subprocess.CompletedProcess[bytes] | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    finalizer_errors: list[str] = []
    try:
        anchor = _capture_process_anchor(process)
        if process.stdout is None or process.stderr is None:
            raise ContractError("dedicated P0 Cargo build pipes are unavailable")
        streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + P0_BUILD_TIMEOUT_SECONDS
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, streams[descriptor])

        leader_exit: Any | None = None
        while selector.get_map() or leader_exit is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ContractError("dedicated P0 Cargo build timed out")
            if leader_exit is None:
                try:
                    _validate_process_group_anchor(anchor)
                    leader_exit = _waitid_without_reap(anchor[0])
                except OSError as exc:
                    raise ContractError(f"P0 Cargo leader wait observation failed: {exc}") from exc
            if not selector.get_map():
                try:
                    time.sleep(min(0.01, remaining))
                except BaseException as exc:
                    raise ContractError(f"P0 Cargo EOF wait failed: {exc}") from exc
                continue
            try:
                events = selector.select(min(remaining, 0.25))
            except BaseException as exc:
                raise ContractError(f"P0 Cargo selector failed: {exc}") from exc
            if not events and leader_exit is not None:
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _mask in events:
                descriptor = int(key.fd)
                try:
                    used = len(buffers["stdout"]) + len(buffers["stderr"])
                    read_size = min(65536, P0_BUILD_OUTPUT_LIMIT_BYTES - used + 1)
                    chunk = os.read(descriptor, read_size)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise ContractError(f"P0 Cargo output read failed: {exc}") from exc
                if not chunk:
                    try:
                        selector.unregister(descriptor)
                    except BaseException as exc:
                        raise ContractError(f"P0 Cargo selector unregister failed: {exc}") from exc
                    continue
                buffers[key.data].extend(chunk)
                if len(buffers["stdout"]) + len(buffers["stderr"]) > P0_BUILD_OUTPUT_LIMIT_BYTES:
                    raise ContractError("dedicated P0 Cargo build output exceeded its bound")

        if leader_exit is None:
            raise ContractError("P0 Cargo leader was not observed exited after EOF")
        _validate_process_group_anchor(anchor)
        members = _process_group_members(anchor)
        other_members = tuple(pid for pid in members if pid != anchor[0])
        if other_members:
            raise ContractError(
                "P0 Cargo process group remained after EOF: "
                + ",".join(str(pid) for pid in other_members)
            )
        try:
            waited_pid, _status = os.waitpid(anchor[0], 0)
        except OSError as exc:
            raise ContractError(f"P0 Cargo leader reap after final group check failed: {exc}") from exc
        if waited_pid != anchor[0]:
            raise ContractError("P0 Cargo leader reap returned an unexpected PID")
        returncode = _exit_code(leader_exit)
        process.returncode = returncode
        result = subprocess.CompletedProcess(
            command,
            returncode,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
        )
    except BaseException as exc:
        primary_error = exc
        try:
            cleanup_errors = _cleanup_process_group(process, anchor)
        except BaseException as cleanup_exc:
            _record_cleanup_error(cleanup_errors, "cleanup coordinator", cleanup_exc)
    finally:
        _close_resource(selector, "selector", finalizer_errors)
        _close_resource(process.stdout, "stdout", finalizer_errors)
        _close_resource(process.stderr, "stderr", finalizer_errors)

    if primary_error is not None:
        diagnostics = cleanup_errors + finalizer_errors
        if diagnostics and hasattr(primary_error, "add_note"):
            primary_error.add_note("P0 Cargo cleanup diagnostics: " + "; ".join(diagnostics))
        raise primary_error
    if finalizer_errors:
        raise ContractError("P0 Cargo finalizer failed: " + "; ".join(finalizer_errors))
    if result is None:
        raise ContractError("P0 Cargo build produced no result")
    return result


def build_artifact(
    *,
    repo: Path = ROOT,
    output_dir: Path,
    candidate: Mapping[str, Any],
    target: str,
    prerequisites: list[dict[str, Any]],
    run_build: bool = True,
) -> dict[str, Any]:
    if target not in ("gfx1030", "gfx1201"):
        raise ContractError("P0 builder target is not canonical")
    validate_candidate(candidate, repo)
    if len(prerequisites) != 5:
        raise ContractError("P0 builder requires exactly five prerequisite records")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = output_dir / P0_BINARY
    sidecar = output_dir / P0_SIDECAR
    artifact_path = output_dir / "rmsnorm-p0-artifact.json"
    if any(path.exists() or path.is_symlink() for path in (binary, sidecar, artifact_path)):
        raise ContractError("P0 builder refuses to overwrite an existing artifact directory")

    build_root = Path(tempfile.mkdtemp(prefix="sllm-p0-cargo-"))
    try:
        if run_build:
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith("CARGO_FEATURE_"):
                    environment.pop(name, None)
            environment.update(p0_build_environment(target))
            environment["CARGO_TARGET_DIR"] = str(build_root)
            completed = _run_bounded_build(
                list(P0_BUILD_COMMAND), cwd=repo, env=environment
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).decode("utf-8", "replace")[-4096:]
                raise ContractError(f"dedicated P0 Cargo build failed: {detail}")
        built_binary = build_root / "release" / P0_BINARY
        _regular(built_binary, "fresh P0 Cargo output", executable=True)
        shutil.copyfile(built_binary, binary)
        binary.chmod(stat.S_IMODE(built_binary.stat().st_mode) | 0o111)
        _regular(binary, "P0 artifact binary", executable=True)
        binary_sha = sha256_file(binary)
        sidecar.write_bytes(f"{binary_sha}  {P0_BINARY}\n".encode("ascii"))
        _regular(sidecar, "P0 artifact sidecar")

        artifact = {
            "schema_version": "rmsnorm-p0-artifact-v1",
            "artifact_id": f"rmsnorm-p0-{target}-{binary_sha}",
            "row_id": f"rmsnorm-p0-{target}",
            "target": target,
            "candidate": dict(candidate),
            "binary": {
                "role": P0_BINARY_ROLE,
                "path": P0_BINARY,
                "sidecar_path": P0_SIDECAR,
                "size_bytes": binary.stat().st_size,
                "sha256": binary_sha,
                "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            },
            "build": {
                "builder": "ci/tools/build_rmsnorm_p0_runtime.py",
                "command": list(P0_BUILD_COMMAND),
                "profile": "release",
                "binary_name": P0_BINARY,
                "output_path": P0_BINARY,
                "fresh_output": True,
                "substitution_rejected": True,
                "environment": p0_build_environment(target),
                "limits": P0_BUILD_LIMITS,
            },
            "source_set": source_set(repo),
            "execution_contract": {
                "public_path": PUBLIC_PATH,
                "kernel_id": 1,
                "kernel_symbol": "rmsnorm.baseline.wave32.v1",
                "device_symbol": "sllm_rmsnorm_baseline_wave32_v1",
                "workgroup_size_x": 256,
                "timing_contract": "rmsnorm-p0-timing-v1",
                "dtype": dict(DTYPE_CONTRACT),
                "producer_status": PRODUCER_STATUS,
            },
            "scope": {
                "selected_backend": "hip",
                "public_rmsnorm_path": True,
                "semantic_op_used": True,
                "model_used": False,
                "hip_only": True,
                "fallback_allowed": False,
                "fallback_used": False,
                "cpu_fallback_used": False,
            },
            "prerequisites": prerequisites,
        }
        artifact_path.write_bytes(canonical_bytes(artifact))
        return validate_artifact(artifact, repo, binary_path=binary)
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prerequisites", required=True, type=Path)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    args = parser.parse_args()
    try:
        prerequisites = read_json(args.prerequisites)
        if not isinstance(prerequisites, list) or not all(isinstance(item, dict) for item in prerequisites):
            raise ContractError("P0 prerequisites must be a JSON array of objects")
        artifact = build_artifact(
            repo=args.repo.resolve(),
            output_dir=args.output_dir,
            candidate=_candidate(args),
            target=args.target,
            prerequisites=prerequisites,
        )
        print(f"P0 artifact: {artifact['artifact_id']}")
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"P0 builder: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
