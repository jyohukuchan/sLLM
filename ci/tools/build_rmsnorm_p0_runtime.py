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


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise ContractError(f"P0 Cargo build process group cannot be inspected: {exc}") from exc
    return True


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate and reap the complete isolated Cargo build process group."""

    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signal_value)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ContractError(f"P0 Cargo build process-group cleanup failed: {exc}") from exc
        deadline = time.monotonic() + P0_BUILD_KILL_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_exists(process.pid):
                if process.poll() is None:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                return
            time.sleep(0.01)
    raise ContractError("P0 Cargo build process group could not be reaped")


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
    if process.stdout is None or process.stderr is None:
        _kill_process_group(process)
        raise ContractError("dedicated P0 Cargo build pipes are unavailable")

    streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + P0_BUILD_TIMEOUT_SECONDS
    try:
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                raise ContractError("dedicated P0 Cargo build timed out")
            events = selector.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _mask in events:
                descriptor = int(key.fd)
                try:
                    chunk = os.read(descriptor, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                buffers[streams[descriptor]].extend(chunk)
                if sum(len(value) for value in buffers.values()) > P0_BUILD_OUTPUT_LIMIT_BYTES:
                    _kill_process_group(process)
                    raise ContractError("dedicated P0 Cargo build output exceeded its bound")
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                _kill_process_group(process)
                raise ContractError("dedicated P0 Cargo build timed out") from exc
    except BaseException:
        if process.poll() is None or _process_group_exists(process.pid):
            _kill_process_group(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(
        command, process.returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


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
