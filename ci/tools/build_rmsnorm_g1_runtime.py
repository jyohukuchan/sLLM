#!/usr/bin/env python3
"""Build a target-isolated semantic RMSNorm G1 runtime artifact.

This module only builds and stages bytes.  It neither launches the runtime nor
returns a PASS result; the controller snapshots its complete output before use.
"""

from __future__ import annotations

import argparse
import array
import base64
import fcntl
import hmac
import io
import json
import os
import selectors
import secrets
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import tarfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from common import ContractError, ROOT  # noqa: E402
import exact_actions  # noqa: E402
import validate_rmsnorm_g1_contracts as contracts  # noqa: E402
import run_rmsnorm_g1_runtime as runner  # noqa: E402


EXPECTED_ROCM_ROOT = Path("/opt/rocm")
EXPECTED_TOOLCHAIN = "1.97.1"
EXPECTED_CODEGEN_FEATURES = "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0"
MAX_BUILD_TIMEOUT_SECONDS = 900.0
MAX_BUILD_RSS_BYTES = 6 * 1024 * 1024 * 1024
BUILD_ADDRESS_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
BUILD_PROCESS_COUNT_LIMIT = 4096
PRIVATE_PREFIX = "sllm-rmsnorm-semantic-g1-"
COMPILER_BROKER_AVAILABLE = True
COMPILER_EXECUTION_PROTOCOL = "parent-owned-exact-action-broker-v1"
COMPILER_BROKER_PROTOCOL = contracts.EXACT_ACTION_PROTOCOL
COMPILER_BROKER_MAX_FRAME = 1024 * 1024
COMPILER_BROKER_MAX_ARGV = 512
COMPILER_BROKER_MAX_ARG_BYTES = 512 * 1024
COMPILER_BROKER_MAX_ENV_BYTES = 512 * 1024
COMPILER_BROKER_MAX_OUTPUT = 256 * 1024
COMPILER_BROKER_MAX_TRANSCRIPT = 64 * 1024 * 1024
COMPILER_BROKER_TIMEOUT_SECONDS = 120.0
COMPILER_BROKER_BIND_TIMEOUT_SECONDS = 5.0
COMPILER_EXEC_READY_TIMEOUT_SECONDS = 5.0
COMPILER_RUNTIME_LD_LIBRARY_PATH = ":".join((
    "/opt/rocm/core-7.14/lib/llvm/lib",
    "/opt/rocm/core-7.14/lib/rocm_sysdeps/lib",
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
))
COMPILER_CLIENT_MAKE_ENVIRONMENT = (
    ("MAKEFLAGS", "s -j1"),
    ("MAKELEVEL", "4"),
    ("MFLAGS", "-s -j1"),
)
COMPILER_EXEC_HELPER_NAME = "compiler-exec-helper"
COMPILER_BROKER_TOKEN_ENV = "SLLM_HIP_COMPILER_BROKER_TOKEN"
COMPILER_BROKER_SOCKET_ENV = "SLLM_HIP_COMPILER_BROKER_SOCKET"
COMPILER_BROKER_SESSION_ENV = "SLLM_HIP_COMPILER_BROKER_SESSION"
COMPILER_BROKER_CLIENT_ENV = "SLLM_HIP_COMPILER_BROKER_CLIENT"
COMPILER_BROKER_CLIENT_SHA_ENV = "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256"
COMPILER_BROKER_CLIENT_FD_ENV = "SLLM_HIP_COMPILER_BROKER_CLIENT_FD"
COMPILER_BROKER_ACTIONS_ENV = "SLLM_HIP_COMPILER_BROKER_ACTIONS_FD"
# The final sealed compiler has a completely parent-derived environment.  Do
# not silently accept any environment variable that can redirect Clang's
# config, resource, include, executable, or library lookup inputs.
COMPILER_FORBIDDEN_INPUT_ENV = frozenset({
    "CLANG_CONFIG_FILE", "CLANG_RESOURCE_DIR", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "GCC_EXEC_PREFIX",
    "COMPILER_PATH", "LIBRARY_PATH", "DEPENDENCIES_OUTPUT",
    "SUNPRO_DEPENDENCIES", "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME", "XDG_DATA_HOME",
})
COMPILER_CLIENT_TEMPLATE = '''#!/usr/bin/python3
import array, base64, hashlib, hmac, json, os, secrets, socket, struct, sys

PROTOCOL = "parent-issued-exact-action-v1"
MAX_FRAME = 1024 * 1024
MAX_ENV_BYTES = 512 * 1024

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def frame(body, token):
    payload = dict(body); payload["mac"] = hmac.new(bytes.fromhex(token), canonical(body), hashlib.sha256).hexdigest()
    data = canonical(payload)
    if len(data) > MAX_FRAME: raise RuntimeError("exact action broker frame is oversized")
    return data

def receive(sock, token):
    payload, control, flags, _ = sock.recvmsg(MAX_FRAME + 1, socket.CMSG_SPACE(16 * struct.calcsize("i")))
    rights = []; unexpected = False
    for level, kind, data in control:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i"); values.frombytes(data[:len(data) - (len(data) % values.itemsize)]); rights.extend(values)
        else: unexpected = True
    for descriptor in set(rights):
        try: os.close(descriptor)
        except OSError: pass
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not payload or len(payload) > MAX_FRAME or unexpected or rights:
        raise RuntimeError("exact action broker response is truncated or carries ancillary data")
    response = json.loads(payload.decode("utf-8")); mac = response.pop("mac", None)
    expected = hmac.new(bytes.fromhex(token), canonical(response), hashlib.sha256).hexdigest()
    if not isinstance(mac, str) or not hmac.compare_digest(mac, expected): raise RuntimeError("exact action broker response authentication failed")
    return response, hashlib.sha256(payload).hexdigest()

def exchange(socket_path, token, body):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        sock.settimeout(120.0); sock.connect(socket_path); sock.send(frame(body, token)); return receive(sock, token)
    finally: sock.close()

def proc_environment():
    result = {}
    with open("/proc/self/environ", "rb") as stream: raw = stream.read(MAX_ENV_BYTES + 1)
    if len(raw) > MAX_ENV_BYTES: raise RuntimeError("exact action client environment is oversized")
    for item in raw.split(b"\\0"):
        if not item: continue
        key, separator, value = item.partition(b"=")
        if not separator: raise RuntimeError("exact action client environment is malformed")
        key_text = key.decode("utf-8")
        if key_text in result: raise RuntimeError("exact action client environment has duplicate keys")
        result[key_text] = value.decode("utf-8")
    return result

def main():
    token = os.environ["SLLM_HIP_COMPILER_BROKER_TOKEN"]; session = os.environ["SLLM_HIP_COMPILER_BROKER_SESSION"]
    socket_path = os.environ["SLLM_HIP_COMPILER_BROKER_SOCKET"]; argv = sys.argv[1:]; environment = proc_environment()
    observation = {"protocol": PROTOCOL, "message_type": "observe", "session": session, "request_nonce": secrets.token_hex(32), "argv": argv, "cwd": os.getcwd(), "environment": environment}
    try:
        issued, _issued_frame_sha256 = exchange(socket_path, token, observation)
        required_issued = {"protocol", "message_type", "session", "request_nonce", "action_manifest"}
        if set(issued) != required_issued or issued["protocol"] != PROTOCOL or issued["message_type"] != "issued" or issued["session"] != session or issued["request_nonce"] != observation["request_nonce"]: raise RuntimeError("exact action issuance response is invalid")
        manifest = issued["action_manifest"]
        if not isinstance(manifest, dict) or manifest.get("argv") != argv or manifest.get("cwd", {}).get("path") != os.getcwd(): raise RuntimeError("exact action issuance does not bind this invocation")
        request = {"protocol": PROTOCOL, "message_type": "execute", "session": session, "request_nonce": secrets.token_hex(32), "observation_nonce": observation["request_nonce"], "action_manifest": manifest}
        response, response_frame_sha256 = exchange(socket_path, token, request)
        required = {"protocol", "message_type", "status", "session", "request_nonce", "source", "client", "action_id", "action_digest", "request_seq", "pid", "starttime", "ppid", "pgrp", "exit_code", "stdout_b64", "stderr_b64", "stdout_sha256", "stderr_sha256", "duration_ns", "timed_out", "crashed", "invocation", "kernel_limits", "exec_identity"}
        if set(response) != required or response["protocol"] != PROTOCOL or response["message_type"] != "result" or response["status"] not in {"ok", "failed"} or response["session"] != session or response["request_nonce"] != request["request_nonce"] or response["action_id"] != manifest.get("action_id") or response["action_digest"] != manifest.get("manifest_digest"): raise RuntimeError("exact action execution response is invalid")
        stdout = base64.b64decode(response["stdout_b64"], validate=True); stderr = base64.b64decode(response["stderr_b64"], validate=True)
        if len(stdout) > 256 * 1024 or len(stderr) > 256 * 1024 or response["stdout_sha256"] != hashlib.sha256(stdout).hexdigest() or response["stderr_sha256"] != hashlib.sha256(stderr).hexdigest(): raise RuntimeError("exact action response output is invalid")
        acknowledgement = {"protocol": PROTOCOL, "message_type": "ack", "session": session, "request_nonce": secrets.token_hex(32), "observation_nonce": observation["request_nonce"], "action_id": manifest["action_id"], "action_digest": manifest["manifest_digest"], "response_frame_sha256": response_frame_sha256}
        acknowledged, _acknowledged_frame_sha256 = exchange(socket_path, token, acknowledgement)
        required_ack = {"protocol", "message_type", "session", "request_nonce", "observation_nonce", "action_id", "action_digest", "response_frame_sha256", "ack_frame_sha256", "acknowledged"}
        if set(acknowledged) != required_ack or acknowledged["protocol"] != PROTOCOL or acknowledged["message_type"] != "acknowledged" or acknowledged["session"] != session or acknowledged["request_nonce"] != acknowledgement["request_nonce"] or acknowledged["observation_nonce"] != observation["request_nonce"] or acknowledged["action_id"] != manifest["action_id"] or acknowledged["action_digest"] != manifest["manifest_digest"] or acknowledged["response_frame_sha256"] != response_frame_sha256 or acknowledged["acknowledged"] is not True: raise RuntimeError("exact action acknowledgement is invalid")
        sys.stdout.buffer.write(stdout); sys.stdout.buffer.flush(); sys.stderr.buffer.write(stderr); sys.stderr.buffer.flush()
        exit_code = int(response["exit_code"])
        if exit_code < 0: os.kill(os.getpid(), -exit_code)
        return exit_code
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError): return 125

raise SystemExit(main())
'''

# This is deliberately a tiny native exec boundary.  It is compiled from
# these reviewed builder bytes with the canonical C++ tool, then made
# read-only before the broker starts.  posix_spawn creates it without a
# Python post-fork path; the helper marks the inherited compiler descriptor
# close-on-exec and uses execveat(AT_EMPTY_PATH), so the final compiler cannot
# retain or address the broker-owned descriptor.
COMPILER_EXEC_HELPER_SOURCE = r'''#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <unistd.h>

#ifndef AT_EMPTY_PATH
#define AT_EMPTY_PATH 0x1000
#endif

static int fail(const char *message) {
    dprintf(STDERR_FILENO, "semantic-g1 compiler exec helper: %s\n", message);
    return 127;
}

int main(int argc, char **argv, char **envp) {
    if (argc < 5 || strncmp(argv[1], "--compiler-fd=", 14) != 0 ||
        strncmp(argv[2], "--cwd=", 6) != 0 || strcmp(argv[3], "--") != 0) {
        return fail("malformed invocation");
    }
    char *end = NULL;
    long descriptor = strtol(argv[1] + 14, &end, 10);
    if (end == argv[1] + 14 || *end != '\0' || descriptor < 3 || descriptor > 1048575) {
        return fail("malformed compiler descriptor");
    }
    if (chdir(argv[2] + 6) != 0) {
        return fail("cannot enter the reviewed compiler cwd");
    }
    // The helper is the last native process boundary before the compiler.
    // Bind the compiler to the broker/controller lifetime without a Python
    // post-fork hook.  The parent check closes the fork/exec death race.
    pid_t parent = getppid();
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) {
        return fail("compiler broker parent died before exec");
    }
    if (fcntl((int)descriptor, F_SETFD, FD_CLOEXEC) != 0) {
        return fail("cannot seal compiler descriptor for exec");
    }
    if (syscall(SYS_execveat, (int)descriptor, "", &argv[4], envp, AT_EMPTY_PATH) != 0) {
        return fail(strerror(errno));
    }
    return fail("unreachable exec return");
}
'''


class BuilderError(ContractError):
    """A fail-closed semantic G1 build or staging violation."""


@dataclass(frozen=True)
class BuildResult:
    row_id: str
    target: str
    output_dir: Path
    cargo_target_dir: Path
    artifact_path: Path
    companion_path: Path
    metadata_path: Path
    artifact_sha256: str
    companion_sha256: str
    metadata_sha256: str
    command: tuple[str, ...]
    compiler_execution: dict[str, Any] | None = None
    runtime_dependency_closure_complete: bool = False


@dataclass(frozen=True)
class _IssuedClientObservation:
    """Authenticated client facts retained between issuance and execution."""

    recipe_key: str
    observation_nonce: str
    argv: tuple[str, ...]
    cwd: str
    environment: dict[str, str]
    client_binding: dict[str, int]
    manifest_digest: str


def _private_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise BuilderError(f"{label} is not an absolute non-symlink directory")
    details = path.stat()
    if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise BuilderError(f"{label} is not private to the current user")


def _new_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise BuilderError(f"cannot create fresh {label}") from exc
    _private_directory(path, label)


def _write_new(path: Path, data: bytes, label: str, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise BuilderError(f"refusing to overwrite {label}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BuilderError(f"cannot write {label}") from exc


def _sidecar(path: Path, label: str) -> None:
    _write_new(path.with_name(path.name + contracts.SIDECAR_SUFFIX), f"{contracts.sha256_file(path)}  {path.name}\n".encode("ascii"), f"{label} sidecar")


def _group_members(pgid: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid:
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return sorted(members)


def _process_facts(pid: int) -> tuple[int, int, int] | None:
    """Return (starttime, parent PID, process group) for one live process."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(") ", 1)[1].split()
        return int(fields[19]), int(fields[1]), int(fields[2])
    except (OSError, IndexError, ValueError):
        return None


def _observe_build_descendants(root_pid: int, pgid: int, tracked: dict[int, int]) -> None:
    """Track every observed build descendant by PID/starttime identity.

    The process group catches ordinary grandchildren.  The parent graph also
    catches a child that calls setsid() and would otherwise leave that group.
    We retain starttimes so a recycled PID can never be mistaken for an old
    build descendant during cleanup.
    """

    facts: dict[int, tuple[int, int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise BuilderError("cannot inspect build descendants") from exc
    for entry in entries:
        if entry.name.isdecimal():
            observed = _process_facts(int(entry.name))
            if observed is not None:
                facts[int(entry.name)] = observed
    roots = {root_pid, *tracked}
    changed = True
    while changed:
        changed = False
        for pid, (starttime, parent_pid, observed_pgrp) in facts.items():
            if pid in roots or parent_pid in roots or observed_pgrp == pgid:
                if pid not in roots:
                    roots.add(pid)
                    changed = True
                tracked.setdefault(pid, starttime)


def _tracked_descendants_alive(tracked: Mapping[int, int]) -> list[int]:
    return sorted(
        pid for pid, starttime in tracked.items()
        if (observed := _process_facts(pid)) is not None and observed[0] == starttime
    )


def _terminate_tracked_descendants(tracked: Mapping[int, int]) -> bool:
    """Terminate identities that escaped the process group, if any."""

    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        survivors = _tracked_descendants_alive(tracked)
        if not survivors:
            return True
        for pid in survivors:
            try:
                os.kill(pid, signal_value)
            except ProcessLookupError:
                continue
            except OSError:
                return False
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and _tracked_descendants_alive(tracked):
            time.sleep(0.02)
    return not _tracked_descendants_alive(tracked)


def _terminate_group(process: subprocess.Popen[bytes], pgid: int, tracked: Mapping[int, int]) -> bool:
    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        if not _group_members(pgid):
            break
        try:
            os.killpg(pgid, signal_value)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and _group_members(pgid):
            time.sleep(0.02)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return False
    return not _group_members(pgid) and _terminate_tracked_descendants(tracked)


def _parent_fd_identities() -> dict[tuple[int, int], int]:
    identities: dict[tuple[int, int], int] = {}
    try:
        entries = list(Path("/proc/self/fd").iterdir())
    except OSError as exc:
        raise BuilderError("cannot inspect parent descriptors for build FD audit") from exc
    for entry in entries:
        try:
            descriptor = int(entry.name)
            details = os.fstat(descriptor)
            identities[(details.st_dev, details.st_ino)] = descriptor
        except (OSError, ValueError):
            continue
    return identities


def _audit_child_fds(pid: int, *, retained_pipe_fds: tuple[int, ...], parent_fds: Mapping[tuple[int, int], int]) -> None:
    """Ensure the direct Cargo child received no parent authority beyond its pipe."""

    try:
        entries = list(Path(f"/proc/{pid}/fd").iterdir())
    except OSError as exc:
        raise BuilderError("cannot inspect direct build child descriptors") from exc
    for entry in entries:
        try:
            descriptor = int(entry.name)
            details = entry.stat()
        except (OSError, ValueError):
            continue
        if descriptor in {0, 1, 2, *retained_pipe_fds}:
            continue
        if (details.st_dev, details.st_ino) in parent_fds:
            raise BuilderError("build child inherited an unapproved controller descriptor")


def _close_rights(data: bytes) -> None:
    """Close every delivered SCM_RIGHTS descriptor exactly once."""

    values = array.array("i")
    values.frombytes(data[: len(data) - (len(data) % values.itemsize)])
    for descriptor in set(values):
        try:
            os.close(descriptor)
        except OSError:
            pass


class _SpawnedCompiler:
    """Waitable native child created by ``os.posix_spawn``."""

    def __init__(self, pid: int, stdout_read: int, stderr_read: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.stdout_read = stdout_read
        self.stderr_read = stderr_read
        self.pidfd = -1

    def pin_identity(self) -> None:
        if not hasattr(os, "pidfd_open"):
            raise BuilderError("compiler exec identity requires Linux pidfd_open")
        try:
            self.pidfd = os.pidfd_open(self.pid, 0)
        except OSError as exc:
            raise BuilderError("compiler exec identity pidfd could not be opened") from exc

    def close_pidfd(self) -> None:
        if self.pidfd >= 0:
            try:
                os.close(self.pidfd)
            except OSError:
                pass
            self.pidfd = -1

    def _poll_wait(self, options: int) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            waited, status = os.waitpid(self.pid, options)
        except ChildProcessError:
            # A containment reaper may have consumed the status.  The broker
            # has already retained the PID/starttime identity and treats this
            # as an exited child, but never fabricates a successful status.
            self.returncode = 125
            return self.returncode
        if waited == 0:
            return None
        self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def poll(self) -> int | None:
        return self._poll_wait(os.WNOHANG)

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        if timeout is None:
            return self._poll_wait(0) or 0
        deadline = time.monotonic() + timeout
        while True:
            result = self._poll_wait(os.WNOHANG)
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("compiler-exec-helper", timeout)
            time.sleep(0.002)


class CompilerBroker:
    """Parent-owned compiler execution broker.

    The sealed compiler descriptor is deliberately never present in the build
    environment or in inherited build descriptors.  Only this object, in the controller
    process, may use it.  Build descendants authenticate a one-request client
    connection and receive only bounded output/status data.
    """

    def __init__(
        self,
        *,
        compiler: contracts.SealedDescriptor,
        client_path: Path,
        exec_helper: Path,
        socket_root: Path,
        allowed_roots: tuple[Path, ...],
        output_roots: tuple[Path, ...] | None = None,
        reviewed_sources: Mapping[str, Mapping[str, Any]] | None = None,
        reviewed_tools: Mapping[str, Mapping[str, Any]] | None = None,
        target: str | None = None,
        expected_environment: Mapping[str, str] | None = None,
        compiler_environment: Mapping[str, str] | None = None,
        action_recipes: Mapping[str, Mapping[str, Any]] | None = None,
        require_complete_recipe_set: bool = False,
    ) -> None:
        if not contracts.descriptor_is_sealed(compiler.fd):
            raise BuilderError("compiler broker requires a fully sealed compiler snapshot")
        self.compiler = compiler
        self.source = dict(compiler.record)
        self.client_path = client_path.resolve(strict=True)
        # The pathname is only a launch handle.  Every accepted request is
        # checked against this sealed byte snapshot and the live pathname is
        # never treated as authority after construction.
        self.client_snapshot = contracts.snapshot_file(self.client_path, None, "compiler broker client")
        self.client_record = dict(self.client_snapshot.record)
        self.exec_helper = exec_helper.resolve(strict=True)
        self.exec_helper_snapshot = contracts.snapshot_file(self.exec_helper, None, "compiler exec helper")
        self.socket_root = socket_root.resolve(strict=True)
        _private_directory(self.socket_root, "compiler broker socket root")
        limiter_expected = (reviewed_tools or {}).get("process_limiter") if isinstance(reviewed_tools, Mapping) else None
        self.process_limiter_snapshot = contracts.snapshot_file(Path(runner.PROCESS_LIMITER), limiter_expected, "compiler process limiter")
        self.exec_helper_record = dict(self.exec_helper_snapshot.record)
        self.allowed_roots = tuple(root.resolve(strict=True) for root in allowed_roots)
        self.output_roots = tuple(root.resolve(strict=True) for root in (output_roots or allowed_roots))
        if not self.output_roots or any(not root.is_dir() for root in self.output_roots):
            raise BuilderError("compiler broker output roots are not closed directories")
        self.reviewed_sources = {
            str(record["path"]): str(record["sha256"])
            for record in (reviewed_sources or {}).values()
            if isinstance(record, Mapping) and isinstance(record.get("path"), str) and isinstance(record.get("sha256"), str)
        }
        self.reviewed_tools = {
            str(record["path"]): str(record["sha256"])
            for record in (reviewed_tools or {}).values()
            if isinstance(record, Mapping) and isinstance(record.get("path"), str) and isinstance(record.get("sha256"), str)
        }
        self.target = target or "gfx1030"
        if self.target not in contracts.TARGETS:
            raise BuilderError("compiler broker target is not a reviewed semantic G1 target")
        if not isinstance(action_recipes, Mapping) or not action_recipes:
            raise BuilderError("compiler broker requires nonempty parent-derived exact action recipes")
        self._action_recipes = {str(key): dict(value) for key, value in action_recipes.items()}
        if len(self._action_recipes) != len(action_recipes):
            raise BuilderError("compiler broker exact action recipe keys are not unique")
        observed_actions: set[tuple[tuple[str, ...], str]] = set()
        for key, recipe in self._action_recipes.items():
            if not key or set(recipe) != {"argv", "cwd", "inputs", "implicit", "response_files", "outputs"}:
                raise BuilderError("compiler broker exact action recipe is not closed")
            argv = recipe["argv"]
            if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item or "\0" in item for item in argv):
                raise BuilderError("compiler broker exact action recipe argv is malformed")
            cwd = Path(str(recipe["cwd"]))
            if not cwd.is_absolute() or cwd.is_symlink() or not cwd.is_dir():
                raise BuilderError("compiler broker exact action recipe cwd is malformed")
            if not self._inside(cwd.resolve(strict=True), self.allowed_roots):
                raise BuilderError("compiler broker exact action recipe cwd is outside the reviewed build roots")
            if not all(isinstance(recipe[name], list) for name in ("inputs", "implicit", "response_files", "outputs")):
                raise BuilderError("compiler broker exact action recipe records are malformed")
            for output in recipe["outputs"]:
                path = Path(str(output))
                if not path.is_absolute() or not self._inside(path.parent.resolve(strict=True), self.output_roots):
                    raise BuilderError("compiler broker exact action recipe output is outside the reviewed output roots")
            identity = (tuple(argv), str(cwd))
            if identity in observed_actions:
                raise BuilderError("compiler broker exact action recipes have an ambiguous observation")
            observed_actions.add(identity)
        requested_compiler_environment = (
            {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
            if compiler_environment is None
            else dict(compiler_environment)
        )
        if not requested_compiler_environment or any(
            not isinstance(key, str) or not isinstance(value, str) or not key or "\0" in key or "\0" in value
            for key, value in requested_compiler_environment.items()
        ):
            raise BuilderError("compiler broker exact compiler environment is malformed")
        broker_environment_names = {
            COMPILER_BROKER_SOCKET_ENV,
            COMPILER_BROKER_TOKEN_ENV,
            COMPILER_BROKER_SESSION_ENV,
            COMPILER_BROKER_CLIENT_ENV,
            COMPILER_BROKER_CLIENT_SHA_ENV,
            COMPILER_BROKER_CLIENT_FD_ENV,
        }
        if broker_environment_names.intersection(requested_compiler_environment):
            raise BuilderError("compiler broker exact compiler environment carries client authentication state")
        self._compiler_spawn_environment = requested_compiler_environment
        self._require_complete_recipe_set = require_complete_recipe_set
        self._expected_recipe_keys = tuple(self._action_recipes)
        compiler_details = os.fstat(self.compiler.fd)
        self._compiler_manifest_identity = {
            **self.source,
            "device": int(compiler_details.st_dev),
            "inode": int(compiler_details.st_ino),
            "seals": int(fcntl.fcntl(self.compiler.fd, fcntl.F_GET_SEALS)),
        }
        self._actions = exact_actions.OneShotBroker()
        self.session = secrets.token_hex(32)
        self.token = secrets.token_hex(32)
        self.socket_path = self.socket_root / f"broker-{secrets.token_hex(12)}.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self.listener.listen(8)
        self.listener.settimeout(0.1)
        self.root_pid: int | None = None
        self.root_starttime: int | None = None
        self.root_pgrp: int | None = None
        self._expected_environment: dict[str, str] | None = None if expected_environment is None else dict(expected_environment)
        self.events: list[dict[str, Any]] = []
        self._issued_observations: dict[str, _IssuedClientObservation] = {}
        self._recipe_by_action: dict[str, str] = {}
        self._pending_deliveries: dict[str, dict[str, Any]] = {}
        self._seen_nonces: set[str] = set()
        self._failure: list[BaseException] = []
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._started = False
        self._closing = False
        self._state = "new"
        self._active = 0
        self._active_compiler_pid: int | None = None
        self._active_compiler: tuple[_SpawnedCompiler, runner.LinuxContainment] | None = None
        self._launching = 0
        self._build_process: subprocess.Popen[bytes] | None = None
        self._build_containment: runner.LinuxContainment | None = None
        self._build_reaped = False
        self._build_quiescence_rounds = 0
        self._build_bound = threading.Event()
        self._accepted_outputs: dict[str, dict[str, Any]] = {}
        self._connections: set[socket.socket] = set()
        self._compiler_lock = threading.Lock()
        self._lifecycle = threading.Condition(self._compiler_lock)
        self._cleanup_lock = threading.Lock()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, name="semantic-g1-compiler-broker", daemon=False)

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure[0] if self._failure else None

    def environment(self) -> dict[str, str]:
        return {
            COMPILER_BROKER_SOCKET_ENV: str(self.socket_path),
            COMPILER_BROKER_TOKEN_ENV: self.token,
            COMPILER_BROKER_SESSION_ENV: self.session,
            COMPILER_BROKER_CLIENT_ENV: self.client_exec_path,
            COMPILER_BROKER_CLIENT_SHA_ENV: self.client_record["sha256"],
            COMPILER_BROKER_CLIENT_FD_ENV: str(self.client_snapshot.fd),
        }

    @property
    def client_exec_path(self) -> str:
        return f"/proc/self/fd/{self.client_snapshot.fd}"

    def child_pass_fds(self) -> tuple[int, ...]:
        return (self.client_snapshot.fd,)

    @property
    def process_limiter_exec_path(self) -> str:
        return f"/proc/self/fd/{self.process_limiter_snapshot.fd}"

    def start(self) -> None:
        if self._started or self._state != "new":
            raise BuilderError("compiler broker was started twice")
        self._started = True
        self._state = "running"
        self._thread.start()

    def bind_build(
        self,
        pid: int,
        pgrp: int,
        environment: Mapping[str, str] | None = None,
        *,
        process: subprocess.Popen[bytes] | None = None,
        containment: runner.LinuxContainment | None = None,
    ) -> None:
        with self._compiler_lock:
            if self._state != "running" or self.root_pid is not None or pid < 1 or pgrp < 1 or pid == os.getpid():
                raise BuilderError("compiler broker build identity was bound twice or outside the open lifecycle")
            facts = _process_facts(pid)
            if facts is None or facts[2] != pgrp:
                raise BuilderError("compiler broker build identity is not a live PID/starttime/process-group binding")
            if environment is not None:
                if self._expected_environment is not None:
                    expected = dict(self._expected_environment)
                    expected.update(self.environment())
                    if dict(environment) != expected:
                        raise BuilderError("compiler broker build registration environment is not exact")
            if self._build_bound.is_set():
                raise BuilderError("compiler broker build identity was registered outside its lifecycle")
            self.root_pid, self.root_starttime, self.root_pgrp = pid, facts[0], pgrp
            self._expected_environment = None if environment is None else dict(environment)
            self._build_process = process
            self._build_containment = containment
            self._build_reaped = False
            self._build_quiescence_rounds = 0
            self._build_bound.set()

    def mark_build_reaped(self) -> None:
        with self._compiler_lock:
            if self.root_pid is None or self._build_containment is None:
                raise BuilderError("compiler broker cannot mark an unregistered build as reaped")
            if self._build_containment.alive():
                raise BuilderError("compiler broker build containment still has live identities")
            if self._build_containment.quiescence_rounds != 3:
                raise BuilderError("compiler broker build containment lacks the required quiescence barrier")
            self._build_quiescence_rounds = self._build_containment.quiescence_rounds
            self._build_reaped = True

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return contracts.canonical_bytes(value)

    def _mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(bytes.fromhex(self.token), self._canonical(body), "sha256").hexdigest()

    def _send(self, conn: socket.socket, body: Mapping[str, Any]) -> str:
        payload = dict(body)
        payload["mac"] = self._mac(body)
        encoded = self._canonical(payload)
        if len(encoded) > COMPILER_BROKER_MAX_FRAME:
            raise BuilderError("compiler broker response exceeds the bounded frame")
        if conn.send(encoded) != len(encoded):
            raise BuilderError("compiler broker response was short-written")
        return contracts.sha256_bytes(encoded)

    def _recv(self, conn: socket.socket) -> tuple[dict[str, Any], str, tuple[int, int, int]]:
        ancillary = socket.CMSG_SPACE(struct.calcsize("3i")) + socket.CMSG_SPACE(16 * struct.calcsize("i"))
        payload, control, flags, _ = conn.recvmsg(COMPILER_BROKER_MAX_FRAME + 1, ancillary)
        credentials: tuple[int, int, int] | None = None
        truncated = bool(flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC))
        bad_ancillary = False
        for level, kind, data in control:
            if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and credentials is None and len(data) == struct.calcsize("3i"):
                credentials = struct.unpack("3i", data)
            elif level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                _close_rights(data)
                bad_ancillary = True
            else:
                bad_ancillary = True
        if truncated or not payload or len(payload) > COMPILER_BROKER_MAX_FRAME:
            raise BuilderError("compiler broker request was truncated or oversized")
        if bad_ancillary:
            raise BuilderError("compiler broker request carried unexpected ancillary data")
        if credentials is None:
            raise BuilderError("compiler broker request did not carry credentials")
        document = contracts.read_json_bytes(payload, "compiler broker request")
        if not isinstance(document, dict) or document.pop("mac", None) != self._mac(document):
            raise BuilderError("compiler broker request authentication failed")
        return document, contracts.sha256_bytes(payload), credentials

    @staticmethod
    def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def _recipe_for_observation(self, argv: list[str], cwd: Path) -> tuple[str, dict[str, Any]]:
        """Derive an exact manifest only from a reviewed fixed action recipe.

        The observation itself is not authority: it receives no compiler work
        until its argv/cwd equals one deterministic parent recipe and the
        parent has sealed the resulting whole action.  The client environment
        is authenticated separately; the manifest environment is the exact
        environment of the final sealed compiler process.
        """

        for recipe_key, recipe in self._action_recipes.items():
            if argv != recipe.get("argv") or cwd != Path(str(recipe.get("cwd"))):
                continue
            inputs = [exact_actions.file_record(Path(str(item["path"])), role=str(item["role"]), label="exact action input") for item in recipe.get("inputs", ())]
            implicit = [exact_actions.implicit_record(role=str(item["role"]), value=bytes(item["bytes"])) for item in recipe.get("implicit", ())]
            response_files = [exact_actions.implicit_record(role=str(item["role"]), value=bytes(item["bytes"])) for item in recipe.get("response_files", ())]
            outputs = [exact_actions.output_record(Path(str(path)), label="exact action") for path in recipe.get("outputs", ())]
            manifest = exact_actions.make_manifest(
                executable=self._compiler_manifest_identity,
                argv0=self._compiler_argv0(), argv=argv, cwd=cwd,
                environment=self._compiler_spawn_environment,
                inputs=inputs, implicit=implicit, response_files=response_files, outputs=outputs, target=self.target,
            )
            return recipe_key, manifest
        raise BuilderError("compiler observation does not equal a parent-derived exact action recipe")

    def _compiler_exec_fd(self) -> int:
        return 198 if self.compiler.fd not in {196, 197, 198} else 199

    def _compiler_argv0(self) -> str:
        return f"/proc/self/fd/{self._compiler_exec_fd()}"

    @staticmethod
    def _proc_environ(pid: int) -> dict[str, str]:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        if len(raw) > COMPILER_BROKER_MAX_ENV_BYTES:
            raise BuilderError("compiler client environment exceeds the broker bound")
        result: dict[str, str] = {}
        for item in raw.split(b"\0"):
            if not item:
                continue
            key, separator, value = item.partition(b"=")
            if not separator:
                raise BuilderError("compiler client environment contains a malformed entry")
            key_text, value_text = key.decode("utf-8"), value.decode("utf-8")
            if key_text in result:
                raise BuilderError("compiler client environment contains a duplicate key")
            result[key_text] = value_text
        return result

    def _is_descendant(self, pid: int) -> bool:
        if self.root_pid is None or self.root_pgrp is None:
            return False
        current = pid
        seen: set[int] = set()
        while current > 1 and current not in seen:
            seen.add(current)
            facts = _process_facts(current)
            if facts is None:
                return False
            _start, parent, pgrp = facts
            if current == self.root_pid or pgrp == self.root_pgrp:
                return True
            current = parent
        return False

    def _validate_observation(self, request: Mapping[str, Any], credentials: tuple[int, int, int]) -> tuple[dict[str, Any], dict[str, str], dict[str, int]]:
        if len(self.events) >= 4096:
            raise BuilderError("compiler broker invocation count exceeded its transcript bound")
        required = {"protocol", "message_type", "session", "request_nonce", "argv", "cwd", "environment"}
        if set(request) != required or request.get("protocol") != COMPILER_BROKER_PROTOCOL or request.get("message_type") != "observe" or request.get("session") != self.session:
            raise BuilderError("exact action observation protocol/state is invalid")
        nonce = request.get("request_nonce")
        if not isinstance(nonce, str) or contracts.SHA256_RE.fullmatch(nonce) is None or nonce in self._seen_nonces:
            raise BuilderError("exact action observation nonce is malformed or replayed")
        argv = request.get("argv")
        if not isinstance(argv, list) or len(argv) > COMPILER_BROKER_MAX_ARGV or any(not isinstance(item, str) or not item or "\x00" in item or len(item.encode()) > 64 * 1024 for item in argv):
            raise BuilderError("exact action observed argv is malformed or oversized")
        if len(self._canonical(argv)) > COMPILER_BROKER_MAX_ARG_BYTES:
            raise BuilderError("exact action observed argv exceeds its bound")
        cwd = request.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute() or "\x00" in cwd:
            raise BuilderError("exact action observed cwd is malformed")
        observed_cwd = Path(f"/proc/{credentials[0]}/cwd").resolve(strict=True)
        if Path(cwd) != observed_cwd or not any(observed_cwd == root or root in observed_cwd.parents for root in self.allowed_roots):
            raise BuilderError("exact action observed cwd is outside the reviewed build roots")
        environment = request.get("environment")
        if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) or "\x00" in k or "\x00" in v for k, v in environment.items()):
            raise BuilderError("exact action observed environment facts are malformed")
        if len(self._canonical(environment)) > COMPILER_BROKER_MAX_ENV_BYTES:
            raise BuilderError("exact action observed environment exceeds its bound")
        observed_environment = self._proc_environ(credentials[0])
        if observed_environment != environment:
            raise BuilderError("exact action observed environment differs from /proc")
        forbidden = {
            "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "LD_DEBUG", "LD_ORIGIN_PATH",
            "PYTHONHOME", "PYTHONPATH", "PYTHONINSPECT", "PYTHONSTARTUP", "PYTHONWARNINGS", "PYTHONUSERBASE",
            "BASH_ENV", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "SLLM_HIP_COMPILER_FD", "SLLM_HIP_COMPILER_WRAPPER_FD",
        }
        if forbidden.intersection(environment):
            raise BuilderError("exact action observation carries a forbidden execution override")
        expected_environment = self._expected_environment
        if expected_environment is not None:
            expected_with_auth = compiler_client_environment(expected_environment)
            expected_with_auth.update(self.environment())
            if environment != expected_with_auth:
                unexpected = sorted(set(environment) ^ set(expected_with_auth))
                changed = sorted(key for key in set(environment) & set(expected_with_auth) if environment[key] != expected_with_auth[key])
                raise BuilderError(f"exact action observation changed the closed build environment: keys={unexpected}, values={changed}")
        if environment.get(COMPILER_BROKER_SOCKET_ENV) != str(self.socket_path) or environment.get(COMPILER_BROKER_TOKEN_ENV) != self.token or environment.get(COMPILER_BROKER_SESSION_ENV) != self.session or environment.get(COMPILER_BROKER_CLIENT_ENV) != self.client_exec_path or environment.get(COMPILER_BROKER_CLIENT_SHA_ENV) != self.client_record["sha256"] or environment.get(COMPILER_BROKER_CLIENT_FD_ENV) != str(self.client_snapshot.fd):
            raise BuilderError("exact action authentication environment is not bound to this session")
        interpreter = Path("/usr/bin/python3")
        live_executable = Path(f"/proc/{credentials[0]}/exe").resolve(strict=True)
        if live_executable != interpreter.resolve(strict=True):
            raise BuilderError("exact action peer executable is not the reviewed Python client interpreter")
        command_line = Path(f"/proc/{credentials[0]}/cmdline").read_bytes().split(b"\0")[:-1]
        try:
            client_fd = int(environment[COMPILER_BROKER_CLIENT_FD_ENV])
            child_client = os.stat(f"/proc/{credentials[0]}/fd/{client_fd}")
            sealed_client = os.fstat(self.client_snapshot.fd)
        except (KeyError, OSError, ValueError) as exc:
            raise BuilderError("exact action client sealed descriptor is unavailable") from exc
        if (child_client.st_dev, child_client.st_ino) != (sealed_client.st_dev, sealed_client.st_ino):
            raise BuilderError("exact action client did not execute the parent-owned sealed object")
        expected_prefix = [str(interpreter).encode(), self.client_exec_path.encode()]
        expected_suffix = [item.encode() for item in argv]
        if command_line[:2] != expected_prefix or command_line[2:] != expected_suffix:
            raise BuilderError("exact action peer command line does not equal the observed invocation")
        binding = contracts.process_binding(credentials[0])
        descendant = self._is_descendant(credentials[0])
        if binding["uid"] != os.getuid() or binding["gid"] != os.getgid() or credentials != (binding["pid"], binding["uid"], binding["gid"]) or not descendant:
            raise BuilderError(
                "exact action peer credentials/process identity are not bound to this build"
            )
        self._seen_nonces.add(nonce)
        return dict(request), dict(environment), binding

    @contextmanager
    def _launch_slot(self):
        with self._lifecycle:
            if self._state != "running":
                raise BuilderError("compiler launch was attempted after broker closing began")
            self._launching += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._launching -= 1
                self._lifecycle.notify_all()

    def _spawn_compiler(self, argv: list[str], environment: Mapping[str, str], cwd: Path, input_view: exact_actions.ImmutableInputView) -> tuple[_SpawnedCompiler, runner.LinuxContainment, dict[str, Any]]:
        with self._launch_slot():
            return self._spawn_compiler_inner(argv, environment, cwd, input_view)

    def _spawn_compiler_inner(self, argv: list[str], environment: Mapping[str, str], cwd: Path, input_view: exact_actions.ImmutableInputView) -> tuple[_SpawnedCompiler, runner.LinuxContainment, dict[str, Any]]:
        containment = runner.LinuxContainment.begin()
        stdout_read = stdout_write = stderr_read = stderr_write = -1
        try:
            stdout_read, stdout_write = os.pipe2(os.O_CLOEXEC)
            stderr_read, stderr_write = os.pipe2(os.O_CLOEXEC)
        except BaseException:
            for descriptor in (stdout_read, stdout_write, stderr_read, stderr_write):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if not containment.restore_after_launch_failure():
                raise BuilderError("compiler broker could not restore containment after pipe allocation failure")
            raise
        exec_fd = self._compiler_exec_fd()
        helper_fd = 197 if self.exec_helper_snapshot.fd not in {exec_fd, 196, 197} else 195
        limiter_fd = 196 if self.process_limiter_snapshot.fd not in {exec_fd, helper_fd, 196} else 194
        actions = [
            (os.POSIX_SPAWN_DUP2, stdout_write, 1),
            (os.POSIX_SPAWN_DUP2, stderr_write, 2),
            (os.POSIX_SPAWN_DUP2, self.compiler.fd, exec_fd),
            (os.POSIX_SPAWN_DUP2, self.exec_helper_snapshot.fd, helper_fd),
            (os.POSIX_SPAWN_DUP2, self.process_limiter_snapshot.fd, limiter_fd),
            (os.POSIX_SPAWN_CLOSE, stdout_read),
            (os.POSIX_SPAWN_CLOSE, stdout_write),
            (os.POSIX_SPAWN_CLOSE, stderr_read),
            (os.POSIX_SPAWN_CLOSE, stderr_write),
        ]
        actions.extend(input_view.spawn_file_actions())
        if exec_fd != self.compiler.fd:
            actions.append((os.POSIX_SPAWN_CLOSE, self.compiler.fd))
        for source_fd in (self.exec_helper_snapshot.fd, self.process_limiter_snapshot.fd):
            if source_fd not in {helper_fd, limiter_fd}:
                actions.append((os.POSIX_SPAWN_CLOSE, source_fd))
        helper_argv = [
            f"/proc/self/fd/{helper_fd}", f"--compiler-fd={exec_fd}", f"--cwd={cwd}", "--",
            self._compiler_argv0(), *argv,
        ]
        command = [
            f"/proc/self/fd/{limiter_fd}", f"--as={BUILD_ADDRESS_LIMIT_BYTES}",
            f"--nproc={BUILD_PROCESS_COUNT_LIMIT}", "--", *helper_argv,
        ]
        try:
            pid = os.posix_spawn(
                f"/proc/self/fd/{limiter_fd}", command, dict(environment),
                file_actions=actions, setsid=True,
            )
        except BaseException:
            for descriptor in (stdout_read, stdout_write, stderr_read, stderr_write):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if not containment.restore_after_launch_failure():
                raise BuilderError("compiler broker could not restore containment after compiler launch failure")
            raise
        os.close(stdout_write)
        os.close(stderr_write)
        process = _SpawnedCompiler(pid, stdout_read, stderr_read)
        try:
            containment.bind_root(process.pid, process.pid)
            process.pin_identity()
            binding = self._wait_for_exec_identity(process, argv, cwd, exec_fd)
            with self._compiler_lock:
                if self._state not in {"running", "closing"}:
                    raise BuilderError("compiler completed exec outside the broker lifecycle")
                self._active_compiler_pid = process.pid
                self._active_compiler = (process, containment)
        except BaseException:
            if not containment.terminate_and_reap(process):
                raise BuilderError("compiler broker failed to clean up an unbound compiler child")
            process.close_pidfd()
            raise
        return process, containment, binding

    def _wait_for_exec_identity(self, process: _SpawnedCompiler, argv: list[str], cwd: Path, exec_fd: int) -> dict[str, Any]:
        """Return facts only after /proc proves the actual sealed object exec'd."""

        sealed = os.fstat(self.compiler.fd)
        expected_command = [self._compiler_argv0(), *argv]
        deadline = time.monotonic() + COMPILER_EXEC_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            facts = _process_facts(process.pid)
            if facts is None:
                if process.poll() is not None:
                    raise BuilderError("compiler exited before sealed exec identity was proven")
                time.sleep(0.002)
                continue
            try:
                executable = os.stat(f"/proc/{process.pid}/exe")
                command_line = Path(f"/proc/{process.pid}/cmdline").read_bytes().split(b"\0")[:-1]
                observed_cwd = Path(f"/proc/{process.pid}/cwd").resolve(strict=True)
                decoded_command = [item.decode("utf-8") for item in command_line]
            except (OSError, UnicodeDecodeError) as exc:
                raise BuilderError(f"compiler exec identity observation failed: {exc}") from exc
            if (executable.st_dev, executable.st_ino) == (sealed.st_dev, sealed.st_ino):
                if process.pidfd < 0 or facts[1] != os.getpid() or decoded_command != expected_command or observed_cwd != cwd or facts[2] != process.pid:
                    raise BuilderError("compiler sealed exec identity/cwd/pgrp/argv mismatch")
                try:
                    signal.pidfd_send_signal(process.pidfd, 0, None, 0)
                except OSError as exc:
                    raise BuilderError("compiler exec identity pidfd no longer names the observed process") from exc
                confirmed_facts = _process_facts(process.pid)
                if confirmed_facts != facts:
                    raise BuilderError("compiler exec identity changed during observation")
                for descriptor in Path(f"/proc/{process.pid}/fd").iterdir():
                    try:
                        details = descriptor.stat()
                    except OSError:
                        continue
                    if (details.st_dev, details.st_ino) == (sealed.st_dev, sealed.st_ino):
                        raise BuilderError("compiler retained the sealed descriptor after successful exec")
                return {
                    "pid": process.pid, "starttime": facts[0], "ppid": facts[1], "pgrp": facts[2],
                    "exe_dev": int(executable.st_dev), "exe_ino": int(executable.st_ino),
                    "sealed_dev": int(sealed.st_dev), "sealed_ino": int(sealed.st_ino),
                    "exe_path": f"/proc/{process.pid}/exe", "argv_sha256": contracts.sha256_json(argv),
                    "cwd": str(cwd), "exec_ready": True,
                }
            if process.poll() is not None:
                raise BuilderError("compiler exited before sealed exec identity was proven")
            time.sleep(0.002)
        raise BuilderError("compiler exec readiness handshake timed out")

    def _run_compiler(self, action_manifest: Mapping[str, Any]) -> dict[str, Any]:
        manifest = contracts._validate_exact_action(action_manifest)
        try:
            manifest = exact_actions.validate_live_manifest(manifest)
        except exact_actions.ExactActionError as exc:
            raise BuilderError(f"exact action identities changed before compiler execution: {exc}") from exc
        # This is the final pathname-to-bytes boundary.  The compiler never
        # receives the validated mutable input path; it receives only the
        # sealed memfd/include view produced after the live identity check.
        try:
            input_view = exact_actions.seal_input_view(manifest)
        except exact_actions.ExactActionError as exc:
            raise BuilderError(f"exact action immutable input view could not be bound: {exc}") from exc
        if manifest["executable"] != self._compiler_manifest_identity or manifest["argv0"] != self._compiler_argv0() or int(fcntl.fcntl(self.compiler.fd, fcntl.F_GET_SEALS)) != manifest["executable"]["seals"]:
            input_view.close()
            raise BuilderError("exact action executable identity or argv0 is not the broker-sealed compiler")
        argv = list(input_view.argv)
        environment = {key: value for key, value in manifest["environment"]}
        cwd = Path(str(manifest["cwd"]["path"]))
        process: _SpawnedCompiler | None = None
        containment: runner.LinuxContainment | None = None
        stdout_read = stderr_read = -1
        stdout = bytearray()
        stderr = bytearray()
        started = time.monotonic_ns()
        binding: dict[str, int] | None = None
        selector: selectors.BaseSelector | None = None
        try:
            # The pipes are returned by _spawn_compiler through the process
            # object's private descriptors only while the broker owns them.
            process, containment, binding = self._spawn_compiler(argv, environment, cwd, input_view)
            stdout_read = int(getattr(process, "stdout_read", -1))
            stderr_read = int(getattr(process, "stderr_read", -1))
            # _spawn_compiler stores the read ends on the process after the
            # identity is captured.  This assignment is intentionally kept
            # here for the type checker and the cleanup proof below.
            if stdout_read < 0 or stderr_read < 0:
                raise BuilderError("compiler broker output pipes are missing")
            selector = selectors.DefaultSelector()
            streams = {stdout_read: stdout, stderr_read: stderr}
            for descriptor in streams:
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ)
            deadline = time.monotonic() + COMPILER_BROKER_TIMEOUT_SECONDS
            while selector.get_map() or process.poll() is None:
                try:
                    containment.assert_rss_within(MAX_BUILD_RSS_BYTES)
                except runner.RunnerError as exc:
                    raise BuilderError("compiler broker Linux containment/RSS proof failed") from exc
                if time.monotonic() >= deadline:
                    raise BuilderError("compiler broker compiler timeout")
                for key, _mask in selector.select(0.02):
                    descriptor = int(key.fileobj)
                    try:
                        chunk = os.read(descriptor, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(descriptor)
                        os.close(descriptor)
                        if descriptor == stdout_read:
                            stdout_read = -1
                        else:
                            stderr_read = -1
                        continue
                    streams[descriptor].extend(chunk)
                    if len(streams[descriptor]) > COMPILER_BROKER_MAX_OUTPUT:
                        raise BuilderError("compiler broker compiler output exceeded its bound")
            process.wait(timeout=1.0)
            if containment.alive():
                raise BuilderError("compiler broker compiler descendant survived")
            with self._cleanup_lock:
                if not containment.terminate_and_reap(process):
                    raise BuilderError("compiler broker compiler cleanup was not proven")
            output_records: list[dict[str, Any]] = []
            for output_spec in manifest["outputs"]:
                output = Path(str(output_spec["path"]))
                if not output.is_file() or output.is_symlink():
                    raise BuilderError(f"compiler broker accepted output was not materialized: {output}")
                record = contracts.file_identity(output, "compiler invocation output")
                record["path"] = str(output)
                output_records.append(record)
                self._accepted_outputs[str(output)] = dict(record)
            return {
                "pid": binding["pid"], "starttime": binding["starttime"], "ppid": binding["ppid"], "pgrp": binding["pgrp"],
                "exit_code": int(process.returncode if process.returncode is not None else 125),
                "stdout_b64": base64.b64encode(bytes(stdout)).decode("ascii"),
                "stderr_b64": base64.b64encode(bytes(stderr)).decode("ascii"),
                "stdout_sha256": contracts.sha256_bytes(bytes(stdout)),
                "stderr_sha256": contracts.sha256_bytes(bytes(stderr)),
                "duration_ns": time.monotonic_ns() - started,
                "status": "ok" if process.returncode == 0 else "failed",
                "timed_out": False, "crashed": process.returncode is not None and process.returncode < 0,
                "invocation": {
                    "action_manifest": manifest,
                    "materialized_outputs": output_records,
                    "sealed_input_view": input_view.transcript(),
                },
                "kernel_limits": {
                    "address_space_bytes": BUILD_ADDRESS_LIMIT_BYTES,
                    "process_count": BUILD_PROCESS_COUNT_LIMIT,
                    "rss_bytes": MAX_BUILD_RSS_BYTES,
                    "enforced_by": str(self.process_limiter_snapshot.record["path"]),
                    "address_space_enforcement": "kernel-prlimit-v1",
                    "process_count_enforcement": "kernel-prlimit-v1",
                    "rss_enforcement": "parent-sampling-only-v1",
                },
                "action_id": manifest["action_id"],
                "action_digest": manifest["manifest_digest"],
                "exec_identity": binding,
            }
        except BaseException:
            if process is not None and containment is not None:
                with self._cleanup_lock:
                    if not containment.terminate_and_reap(process):
                        raise BuilderError("compiler broker failed to clean up compiler after an execution error")
            raise
        finally:
            with self._compiler_lock:
                self._active_compiler_pid = None
                self._active_compiler = None
            if selector is not None:
                selector.close()
            input_view.close()
            for descriptor in (stdout_read, stderr_read):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if process is not None:
                process.close_pidfd()

    def _handle(self, conn: socket.socket) -> None:
        with self._compiler_lock:
            if self._state != "running":
                conn.close()
                return
            self._active += 1
        started = time.monotonic_ns()
        try:
            conn.settimeout(COMPILER_BROKER_TIMEOUT_SECONDS)
            if not self._build_bound.wait(COMPILER_BROKER_BIND_TIMEOUT_SECONDS):
                raise BuilderError("compiler broker client arrived before atomic build registration")
            with self._compiler_lock:
                if self._state != "running" or self.root_pid is None:
                    raise BuilderError("compiler broker request arrived outside the registered build lifetime")
            request, request_frame_sha, credentials = self._recv(conn)
            if request.get("message_type") == "observe":
                observation, environment, client_binding = self._validate_observation(request, credentials)
                recipe_key, manifest = self._recipe_for_observation(list(observation["argv"]), Path(str(observation["cwd"])))
                issued, newly_issued = self._actions.issue(recipe_key, manifest)
                action_id = str(issued["action_id"])
                if not newly_issued or action_id in self._issued_observations:
                    raise BuilderError("exact action recipe was observed more than once")
                self._issued_observations[action_id] = _IssuedClientObservation(
                    recipe_key=recipe_key,
                    observation_nonce=str(observation["request_nonce"]),
                    argv=tuple(str(value) for value in observation["argv"]),
                    cwd=str(observation["cwd"]),
                    environment=dict(environment),
                    client_binding=dict(client_binding),
                    manifest_digest=str(issued["manifest_digest"]),
                )
                self._recipe_by_action[action_id] = recipe_key
                self._send(conn, {
                    "protocol": COMPILER_BROKER_PROTOCOL, "message_type": "issued", "session": self.session,
                    "request_nonce": observation["request_nonce"], "action_manifest": issued,
                })
                return
            if request.get("message_type") == "execute":
                required_execute = {"protocol", "message_type", "session", "request_nonce", "observation_nonce", "action_manifest"}
                if set(request) != required_execute or request.get("protocol") != COMPILER_BROKER_PROTOCOL or request.get("session") != self.session:
                    raise BuilderError("exact action execution request protocol/state is invalid")
                manifest = contracts._validate_exact_action(request["action_manifest"])
                issuance = self._issued_observations.get(str(manifest["action_id"]))
                if issuance is None or issuance.manifest_digest != manifest["manifest_digest"] or request.get("observation_nonce") != issuance.observation_nonce:
                    raise BuilderError("exact action execution is not bound to its stored client observation")
                action_request = {
                    "protocol": COMPILER_BROKER_PROTOCOL, "message_type": "observe", "session": self.session,
                    "request_nonce": request["request_nonce"], "argv": list(issuance.argv),
                    "cwd": issuance.cwd, "environment": issuance.environment,
                }
                observation, client_environment, client_binding = self._validate_observation(action_request, credentials)
                if (
                    tuple(observation["argv"]) != issuance.argv
                    or str(observation["cwd"]) != issuance.cwd
                    or client_environment != issuance.environment
                    or client_binding != issuance.client_binding
                ):
                    raise BuilderError("exact action execution client no longer equals the stored issuance observation")
                consumed = self._actions.consume(manifest)
                result = self._run_compiler(consumed)
                response_body = {
                    "protocol": COMPILER_BROKER_PROTOCOL, "message_type": "result", "status": result["status"],
                    "session": self.session, "request_nonce": request["request_nonce"],
                    "source": self.source, "client": self.client_record, "request_seq": len(self.events),
                    **result,
                }
                response_frame_sha = self._send(conn, response_body)
                action_id = str(consumed["action_id"])
                self._pending_deliveries[action_id] = {
                    "sequence": len(self.events), "request_nonce": request["request_nonce"],
                    "observation_nonce": issuance.observation_nonce,
                    "client_observation": {
                        "observation_nonce": issuance.observation_nonce,
                        "argv": list(issuance.argv), "cwd": issuance.cwd,
                        "environment_sha256": contracts.sha256_json(issuance.environment),
                        "client_binding": client_binding,
                    },
                    "client_binding": client_binding, "action_id": action_id,
                    "action_digest": consumed["manifest_digest"], "action_manifest": consumed,
                    "request_frame_sha256": request_frame_sha, "response_frame_sha256": response_frame_sha,
                    "compiler_source_sha256": self.source["sha256"], "compiler": result,
                    "started_at_ns": started, "finished_at_ns": time.monotonic_ns(), "consumed": True,
                }
                return
            required_ack = {"protocol", "message_type", "session", "request_nonce", "observation_nonce", "action_id", "action_digest", "response_frame_sha256"}
            if set(request) != required_ack or request.get("protocol") != COMPILER_BROKER_PROTOCOL or request.get("message_type") != "ack" or request.get("session") != self.session:
                raise BuilderError("exact action acknowledgement protocol/state is invalid")
            action_id = request.get("action_id")
            if not isinstance(action_id, str):
                raise BuilderError("exact action acknowledgement action identity is malformed")
            pending = self._pending_deliveries.get(action_id)
            issuance = self._issued_observations.get(action_id)
            if pending is None or issuance is None or request.get("action_digest") != pending["action_digest"] or request.get("response_frame_sha256") != pending["response_frame_sha256"] or request.get("observation_nonce") != issuance.observation_nonce:
                raise BuilderError("exact action acknowledgement is not bound to a delivered response")
            acknowledgement_observation = {
                "protocol": COMPILER_BROKER_PROTOCOL, "message_type": "observe", "session": self.session,
                "request_nonce": request["request_nonce"], "argv": list(issuance.argv),
                "cwd": issuance.cwd, "environment": issuance.environment,
            }
            _observation, _environment, client_binding = self._validate_observation(acknowledgement_observation, credentials)
            if client_binding != issuance.client_binding:
                raise BuilderError("exact action acknowledgement client is not the issued client observation")
            event = {**pending, "ack_frame_sha256": request_frame_sha, "acknowledged": True}
            if len(self._canonical([*self.events, event])) > COMPILER_BROKER_MAX_TRANSCRIPT:
                raise BuilderError("compiler broker transcript exceeded its bounded memory budget")
            self.events.append(event)
            del self._pending_deliveries[action_id]
            self._send(conn, {
                "protocol": COMPILER_BROKER_PROTOCOL, "message_type": "acknowledged", "session": self.session,
                "request_nonce": request["request_nonce"], "observation_nonce": issuance.observation_nonce,
                "action_id": action_id, "action_digest": pending["action_digest"],
                "response_frame_sha256": pending["response_frame_sha256"], "ack_frame_sha256": request_frame_sha,
                "acknowledged": True,
            })
        except BaseException as exc:
            with self._lock:
                self._failure.append(exc)
        finally:
            with self._compiler_lock:
                self._active -= 1
                self._connections.discard(conn)
            conn.close()

    def _serve(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self.listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with self._compiler_lock:
                    if self._state != "running":
                        conn.close()
                        continue
                    self._connections.add(conn)
                self._handle(conn)
        except BaseException as exc:
            with self._lock:
                self._failure.append(exc)
        finally:
            self._closed.set()

    def close(self, *, build_reaped: bool, validate: bool = True) -> dict[str, Any]:
        if self._state == "closed":
            return self.transcript(validate=validate)
        if self._state == "aborted":
            raise BuilderError("compiler broker cannot close after an aborted lifecycle")
        with self._lifecycle:
            self._closing = True
            self._state = "closing"
            while self._launching:
                self._lifecycle.wait()
        self._build_bound.set()
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass
        with self._compiler_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        self._thread.join()
        if self._thread.is_alive():
            raise BuilderError("compiler broker did not close")
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise BuilderError("compiler broker socket cleanup failed") from exc
        root_facts = None if self.root_pid is None else _process_facts(self.root_pid)
        root_alive = self.root_starttime is not None and root_facts is not None and root_facts[0] == self.root_starttime
        group_members = [] if self.root_pgrp is None else _group_members(self.root_pgrp)
        if not build_reaped or not self._build_reaped or self._active or self._launching or self.failure is not None or root_alive or group_members or self._active_compiler is not None or self._pending_deliveries:
            raise BuilderError("compiler broker closed without a clean build lifetime")
        with self._compiler_lock:
            self._state = "closed"
        self._actions.terminal()
        self.client_snapshot.close()
        self.exec_helper_snapshot.close()
        self.process_limiter_snapshot.close()
        self._closed.set()
        return self.transcript(validate=validate)

    def abort(self) -> None:
        with self._lifecycle:
            if self._state == "closed":
                return
            self._state = "closing"
            while self._launching:
                self._lifecycle.wait()
        self._build_bound.set()
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass
        with self._compiler_lock:
            connections = tuple(self._connections)
            active_compiler = self._active_compiler
            build_process = self._build_process
            build_containment = self._build_containment
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        if active_compiler is not None:
            process, containment = active_compiler
            with self._cleanup_lock:
                if not containment.terminate_and_reap(process):
                    with self._lock:
                        self._failure.append(BuilderError("compiler broker abort could not reap the active compiler"))
        if build_process is not None and build_containment is not None:
            with self._cleanup_lock:
                if not build_containment.terminate_and_reap(build_process):
                    with self._lock:
                        self._failure.append(BuilderError("compiler broker abort could not reap the build tree"))
        if self._started:
            self._thread.join()
        try:
            self.socket_path.unlink()
        except OSError:
            pass
        self.client_snapshot.close()
        self.exec_helper_snapshot.close()
        self.process_limiter_snapshot.close()
        with self._compiler_lock:
            self._state = "aborted"
        self._actions.terminal()
        if (
            self._active
            or self._active_compiler is not None
            or (self._build_containment is not None and self._build_containment.alive())
            or (self.root_pgrp is not None and _group_members(self.root_pgrp))
        ):
            raise BuilderError("compiler broker abort left a live build-tree identity")

    def transcript(self, *, validate: bool = True) -> dict[str, Any]:
        if not self._closing or self._active:
            raise BuilderError("compiler broker transcript requested before closure")
        events = [dict(event) for event in self.events]
        actions = []
        for action in self._actions.transcript():
            action_id = str(action["action_id"])
            recipe_key = self._recipe_by_action.get(action_id)
            if recipe_key is None:
                raise BuilderError("compiler broker action has no parent recipe binding")
            actions.append({**action, "recipe_key": recipe_key})
        if self._require_complete_recipe_set:
            issued_keys = {str(item["recipe_key"]) for item in actions}
            consumed_ids = {str(event["action_id"]) for event in events}
            if (
                issued_keys != set(self._expected_recipe_keys)
                or len(actions) != len(self._expected_recipe_keys)
                or any(item["state"] != "consumed" for item in actions)
                or consumed_ids != {str(item["action_id"]) for item in actions}
                or len(events) != len(self._expected_recipe_keys)
            ):
                raise BuilderError("semantic G1 did not issue and consume every reviewed compiler recipe exactly once")
        closure = {
            "state": "closed", "build_root_pid": self.root_pid, "build_root_starttime": self.root_starttime, "build_root_pgrp": self.root_pgrp,
            "build_tree_reaped": bool(self._build_reaped), "listener_closed": True, "active_requests": 0,
            "quiescence_rounds": self._build_quiescence_rounds, "state_machine": "new-running-closing-closed-v1",
            "request_count": len(events), "last_sequence": len(events) - 1,
            "events_sha256": contracts.sha256_json(events),
        }
        record = {
            "protocol": COMPILER_EXECUTION_PROTOCOL, "event_protocol": COMPILER_BROKER_PROTOCOL,
            "source": self.source, "client": self.client_record, "exec_helper": self.exec_helper_record,
            "session": self.session,
            "request_count": len(events), "events_sha256": contracts.sha256_json(events),
            "expected_recipe_keys": list(self._expected_recipe_keys) if self._require_complete_recipe_set else [],
            "closure": closure, "actions": actions, "events": events,
        }
        if validate:
            try:
                contracts._validate_serialized_compiler_execution(record)
            except contracts.EvidenceError as exc:
                raise BuilderError("compiler broker produced an invalid closed transcript") from exc
        return record


def _run(
    argv: list[str], *, cwd: Path, env: Mapping[str, str], timeout: float,
    compiler_source: Mapping[str, Any] | None = None,
    broker: CompilerBroker | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a bounded build, allowing only a brokered build or fixed tools."""

    if timeout <= 0 or timeout > MAX_BUILD_TIMEOUT_SECONDS or not argv or not Path(argv[0]).is_absolute():
        raise BuilderError("builder command/timeout is outside the fixed bounded contract")
    if compiler_source is not None:
        if broker is None or broker.source != dict(compiler_source):
            raise BuilderError("compiler execution requires the parent-owned broker and exact sealed source")
    elif Path(argv[0]) not in {
        EXPECTED_ROCM_ROOT / "lib/llvm/bin/llvm-objcopy",
        EXPECTED_ROCM_ROOT / "lib/llvm/bin/clang-offload-bundler",
    }:
        raise BuilderError("FAIL-CLOSED: an unbrokered build/compiler command is forbidden")
    stage_env = dict(env)
    if broker is not None:
        stage_env.update(broker.environment())
    parent_fds = _parent_fd_identities()
    process: subprocess.Popen[bytes] | None = None
    containment: runner.LinuxContainment | None = None
    stdout = bytearray()
    stderr = bytearray()
    cleanup_ok = False
    broker_closed = False
    try:
        containment = runner.LinuxContainment.begin()
        if not os.path.isfile(runner.PROCESS_LIMITER) or not os.access(runner.PROCESS_LIMITER, os.X_OK):
            raise BuilderError("semantic G1 requires the fixed build prlimit containment primitive")
        limiter_path = broker.process_limiter_exec_path if broker is not None else str(runner.PROCESS_LIMITER)
        pass_fds = (broker.process_limiter_snapshot.fd, *broker.child_pass_fds()) if broker is not None else ()
        launch_argv = [limiter_path, f"--as={BUILD_ADDRESS_LIMIT_BYTES}", f"--nproc={BUILD_PROCESS_COUNT_LIMIT}", "--", *argv]
        process = subprocess.Popen(
            launch_argv, cwd=cwd, env=stage_env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            close_fds=True, pass_fds=pass_fds,
        )
        pgid = process.pid
        containment.bind_root(process.pid, pgid)
        if broker is not None:
            broker.bind_build(process.pid, pgid, stage_env, process=process, containment=containment)
        _audit_child_fds(process.pid, retained_pipe_fds=pass_fds, parent_fds=parent_fds)
        if process.stdout is None or process.stderr is None:
            raise BuilderError("builder diagnostic pipes are missing")
        selector = selectors.DefaultSelector()
        streams = {process.stdout.fileno(): stdout, process.stderr.fileno(): stderr}
        try:
            for descriptor in streams:
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            while selector.get_map():
                try:
                    containment.assert_rss_within(MAX_BUILD_RSS_BYTES)
                except runner.RunnerError as exc:
                    raise BuilderError("builder Linux containment/RSS proof failed") from exc
                if broker is not None and broker.failure is not None:
                    raise BuilderError("compiler broker failed") from broker.failure
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BuilderError(f"builder command timed out: {argv[0]}")
                for key, _mask in selector.select(min(remaining, 0.05)):
                    descriptor = int(key.fileobj)
                    try:
                        chunk = os.read(descriptor, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(descriptor)
                        continue
                    streams[descriptor].extend(chunk)
                    if len(streams[descriptor]) > contracts.MAX_OUTPUT * 16:
                        raise BuilderError("builder command output exceeded its bounded limit")
        finally:
            selector.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise BuilderError("builder direct child did not exit after pipe closure") from exc
        if broker is not None:
            with broker._cleanup_lock:
                build_cleanup_ok = containment.terminate_and_reap(process)
        else:
            build_cleanup_ok = containment.terminate_and_reap(process)
        if not build_cleanup_ok:
            raise BuilderError("builder could not prove full Linux descendant containment/reaping")
        cleanup_ok = True
        if broker is not None:
            broker.mark_build_reaped()
            broker.close(build_reaped=True)
            broker_closed = True
        result = subprocess.CompletedProcess(argv, process.returncode, bytes(stdout), bytes(stderr))
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace")[-1000:]
            raise BuilderError(f"builder command failed ({result.returncode}): {detail}")
        return result
    except (OSError, runner.RunnerError) as exc:
        raise BuilderError(f"builder command could not complete: {argv[0]}") from exc
    finally:
        if process is not None:
            if not cleanup_ok:
                if containment is None or not containment.terminate_and_reap(process):
                    # This branch deliberately cannot turn a failed build into
                    # a normal return; the surrounding exception is retained.
                    cleanup_ok = False
            runner.close_process_streams(process)
        elif containment is not None:
            cleanup_ok = containment.restore_after_launch_failure() and cleanup_ok
        if broker is not None and not broker_closed:
            broker.abort()


def _validate_inherited_environment(target: str) -> None:
    expected = {
        "ROCM_PATH": str(EXPECTED_ROCM_ROOT),
        "HIP_PATH": str(EXPECTED_ROCM_ROOT),
        "SLLM_HIP_COMPILER": str(EXPECTED_ROCM_ROOT / "bin/amdclang++"),
        "CMAKE_HIP_ARCHITECTURES": target,
        "SLLM_HIP_CODEGEN_FEATURES": EXPECTED_CODEGEN_FEATURES,
        "SLLM_ENABLE_HIP_RUNTIME": "0",
        "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "1",
        "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
    }
    for name, value in expected.items():
        if name in os.environ and os.environ[name] != value:
            raise BuilderError(f"inherited {name} disagrees with the semantic G1 tuple")
    forbidden = {
        "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER",
        "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP", "CXX", "CC", "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "RUSTUP_HOME", "CARGO_HOME",
    }
    active = sorted(name for name, value in os.environ.items() if value and (name in forbidden or name.startswith("CARGO_TARGET_") or name.startswith("CARGO_PROFILE_")))
    if active:
        raise BuilderError(f"inherited build override is forbidden: {', '.join(active)}")


def build_environment(target: str, cargo_target_dir: Path, native_build_dir: Path) -> dict[str, str]:
    _validate_inherited_environment(target)
    tools = contracts.canonical_build_tools()
    machine = os.uname().machine
    if machine != "x86_64":
        raise BuilderError("semantic G1 fixed Rust toolchain is only defined for x86_64 Linux")
    rustc = contracts.CANONICAL_RUSTUP_HOME / "toolchains" / f"{EXPECTED_TOOLCHAIN}-x86_64-unknown-linux-gnu" / "bin" / "rustc"
    try:
        rustc = rustc.resolve(strict=True)
    except OSError as exc:
        raise BuilderError("fixed Rust 1.97.1 compiler is unavailable") from exc
    if not rustc.is_file() or not os.access(rustc, os.X_OK):
        raise BuilderError("fixed Rust 1.97.1 compiler is not executable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(contracts.CANONICAL_RUSTUP_HOME.parent),
        "LC_ALL": "C",
        "LANG": "C",
        "RUSTUP_HOME": str(contracts.CANONICAL_RUSTUP_HOME),
        "CARGO_HOME": str(contracts.CANONICAL_CARGO_HOME),
        "RUSTUP_TOOLCHAIN": EXPECTED_TOOLCHAIN,
        "RUSTC": str(rustc),
        "CXX": str(contracts.CANONICAL_CXX),
        "ROCM_PATH": str(EXPECTED_ROCM_ROOT),
        "HIP_PATH": str(EXPECTED_ROCM_ROOT),
        "SLLM_HIP_COMPILER": str(EXPECTED_ROCM_ROOT / "bin/amdclang++"),
        "CMAKE_HIP_ARCHITECTURES": target,
        "SLLM_HIP_CODEGEN_FEATURES": EXPECTED_CODEGEN_FEATURES,
        "SLLM_ENABLE_HIP_RUNTIME": "0",
        "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "1",
        "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
        # This gates the parent-owned sealed compiler broker in the shared
        # build registration.  Generic/H3 compilation deliberately remains on
        # its existing direct compiler path; only a G1 controller-owned build
        # may require the broker session below.
        "SLLM_SEMANTIC_G1_AUTHORITY": "1",
        "CARGO_TARGET_DIR": str(cargo_target_dir),
        "SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR": str(native_build_dir),
    }
    if Path(contracts.EXPECTED_COMMAND[0]).resolve(strict=True) != tools["cargo"]:
        raise BuilderError("semantic G1 matrix does not bind the canonical Cargo executable")
    return environment


def compiler_spawn_environment(build_environment: Mapping[str, str], compiler_logical_path: str) -> dict[str, str]:
    """Return the reviewed compiler environment before any client auth is added.

    The broker client requires session credentials in its own environment.  The
    sealed compiler must never receive those credentials, and its environment
    must be recorded verbatim in the parent-issued action manifest.  Keeping
    this derivation outside ``_spawn_compiler_inner`` makes the later launch a
    byte-for-byte use of the manifest rather than a hidden transformation.
    """

    result = dict(build_environment)
    forbidden = sorted(name for name in COMPILER_FORBIDDEN_INPUT_ENV if name in result)
    if forbidden:
        raise BuilderError(
            "semantic G1 compiler environment permits mutable input configuration: "
            + ", ".join(forbidden)
        )
    for name in (
        COMPILER_BROKER_SOCKET_ENV,
        COMPILER_BROKER_TOKEN_ENV,
        COMPILER_BROKER_SESSION_ENV,
        COMPILER_BROKER_CLIENT_ENV,
        COMPILER_BROKER_CLIENT_SHA_ENV,
        COMPILER_BROKER_CLIENT_FD_ENV,
    ):
        result.pop(name, None)
    result.pop("SLLM_HIP_COMPILER", None)
    # The client/build needs HOME for Rust tooling, but a compiler HOME can
    # select user configuration outside the reviewed closure.  The compiler
    # has no HOME/XDG configuration input at all.
    result.pop("HOME", None)
    result["SLLM_HIP_COMPILER_LOGICAL"] = compiler_logical_path
    result["LD_LIBRARY_PATH"] = COMPILER_RUNTIME_LD_LIBRARY_PATH
    if not result or any(not isinstance(key, str) or not isinstance(value, str) or not key or "\0" in key or "\0" in value for key, value in result.items()):
        raise BuilderError("semantic G1 compiler spawn environment is malformed")
    return result


def compiler_client_environment(build_environment: Mapping[str, str]) -> dict[str, str]:
    """Bind the fixed GNU Make recursion facts seen by HIP compiler actions."""

    result = dict(build_environment)
    make_names = {name for name, _value in COMPILER_CLIENT_MAKE_ENVIRONMENT}
    if make_names.intersection(result):
        raise BuilderError("semantic G1 build environment already carries GNU Make recursion state")
    result.update(COMPILER_CLIENT_MAKE_ENVIRONMENT)
    return result


def _semantic_exact_action_recipes(build_repo: Path, native_build_dir: Path, target: str) -> dict[str, dict[str, Any]]:
    """Declare the only two HIP actions semantic G1 can ever issue.

    CMake configure is compiler-less in this mode.  These actions correspond
    byte-for-byte to the reviewed custom commands in ``native/hip/CMakeLists``;
    future native features extend this declarative list rather than broker
    authorization code.
    """

    include = build_repo / "include"
    source_root = build_repo / "native/hip/src"
    cmake = build_repo / "native/hip/CMakeLists.txt"
    build_rs = build_repo / "crates/sllm-hip-sys/build.rs"
    common = [
        "-D__HIP_ROCclr__=1", "-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=1", "-O3", "-DNDEBUG",
        "-std=gnu++17", "-fPIC", "-I", str(include), "-I", str(source_root),
        f"--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64",
        "-pthread", "-x", "hip", "-c",
    ]
    recipes: dict[str, dict[str, Any]] = {}
    for role, source_name, headers in (
        ("public-runtime", "public_runtime.hip.cpp", ("public_runtime_internal.hpp", "rmsnorm_api.hpp")),
        ("rmsnorm-kernel", "rmsnorm_kernel.hip.cpp", ("rmsnorm_kernel_internal.hpp",)),
    ):
        source = source_root / source_name
        output = native_build_dir / f"{source_name.removesuffix('.cpp')}.o"
        argv = [*common, str(source), "-o", str(output)]
        recipe_key = exact_actions.sha256({"role": role, "argv": argv, "cwd": str(native_build_dir), "target": target})
        recipes[recipe_key] = {
            "argv": argv,
            "cwd": str(native_build_dir),
            "inputs": [
                {"role": "translation-unit", "path": str(source)},
                *({"role": "native-header", "path": str(source_root / header)} for header in headers),
                {"role": "public-header", "path": str(include / "sllm/hip.h")},
            ],
            "implicit": [
                {"role": "cmake-custom-action", "bytes": cmake.read_bytes()},
                {"role": "cargo-build-script", "bytes": build_rs.read_bytes()},
                {"role": "fixed-cmake-configuration", "bytes": exact_actions.canonical_bytes({
                    "cmake_hip_architectures": target,
                    "cmake_hip_compiler_logical": str(EXPECTED_ROCM_ROOT / "bin/amdclang++"),
                    "semantic_g1_authority": True,
                    "hip_codegen_features": EXPECTED_CODEGEN_FEATURES,
                    "public_hip_runtime": True,
                })},
            ],
            "response_files": [{"role": "response-files", "bytes": b""}],
            "outputs": [str(output)],
        }
    return recipes


def _dependency_paths_from_make_output(output: bytes, *, label: str) -> list[Path]:
    """Parse the bounded ``amdclang++ -M`` dependency syntax fail-closed."""

    try:
        lines = output.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError as exc:
        raise BuilderError(f"{label} dependency output is not UTF-8") from exc
    logical_lines: list[str] = []
    pending = ""
    for line in lines:
        if not line or line.startswith("# __CLANG_OFFLOAD_BUNDLE"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical_lines.append(pending + line)
        pending = ""
    if pending or len(logical_lines) != 1 or ":" not in logical_lines[0]:
        raise BuilderError(f"{label} dependency output is not one complete make rule")
    _target, raw_paths = logical_lines[0].split(":", 1)
    paths = raw_paths.split()
    if not paths or any("\\" in value or not Path(value).is_absolute() for value in paths):
        raise BuilderError(f"{label} dependency paths are not closed absolute paths")
    return [Path(value) for value in paths]


def _compiler_resource_directory(output: bytes) -> Path:
    """Accept only the sealed driver's one absolute ROCm resource directory."""

    try:
        value = output.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise BuilderError("compiler resource directory output is not UTF-8") from exc
    if not value.endswith("\n") or value.count("\n") != 1:
        raise BuilderError("compiler resource directory output is not one path")
    path = Path(value[:-1])
    if not path.is_absolute() or path.is_symlink():
        raise BuilderError("compiler resource directory is not an absolute non-symlink path")
    try:
        resolved = path.resolve(strict=True)
        rocm = EXPECTED_ROCM_ROOT.resolve(strict=True)
    except OSError as exc:
        raise BuilderError("compiler resource directory is unavailable") from exc
    if not resolved.is_dir() or (resolved != rocm and rocm not in resolved.parents):
        raise BuilderError("compiler resource directory escapes the reviewed ROCm tree")
    return resolved


def _sealed_compiler_probe(
    compiler: contracts.SealedDescriptor,
    exec_helper: Path,
    compiler_environment: Mapping[str, str],
    cwd: Path,
    argv: list[str],
) -> subprocess.CompletedProcess[bytes]:
    """Run a bounded discovery action with the same sealed driver semantics."""

    if exec_helper.is_symlink() or not exec_helper.is_file() or not os.access(exec_helper, os.X_OK):
        raise BuilderError("compiler probe exec helper is unavailable")
    return subprocess.run(
        [
            str(runner.PROCESS_LIMITER),
            f"--as={BUILD_ADDRESS_LIMIT_BYTES}",
            f"--nproc={BUILD_PROCESS_COUNT_LIMIT}",
            "--",
            str(exec_helper),
            f"--compiler-fd={compiler.fd}",
            f"--cwd={cwd}",
            "--",
            str(compiler.record["path"]),
            *argv,
        ],
        cwd=cwd,
        env=dict(compiler_environment),
        pass_fds=(compiler.fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=COMPILER_BROKER_TIMEOUT_SECONDS,
    )


def _compiler_input_closures(
    compiler: contracts.SealedDescriptor,
    exec_helper: Path,
    compiler_environment: Mapping[str, str],
    recipes: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Capture all mutable final-compiler inputs before action issuance.

    The parent uses the sealed compiler only for dependency discovery with the
    exact final compiler environment and fixed compile flags.  ``-M`` reports
    the complete preprocessor header closure; the sealed driver's resource
    directory is recursively bound to cover Clang configuration/builtin input;
    all AMGPU device bitcode and the driver's dynamic-loader closure are added
    because code generation may consume them without a C/C++ include edge.
    The final compiler has no HOME/XDG/config/include-search override inputs.
    Every discovered file is then inserted into the action manifest and
    opened, byte-checked, and copied into the sealed immutable input view
    immediately before final spawn.  A changed, missing, or symlinked closure
    member fails closed instead of being called an exact input.
    """

    runtime = contracts.runtime_dependency_closure(Path(str(compiler.record["path"])))
    runtime_objects = runtime.get("objects")
    if not isinstance(runtime_objects, list) or not runtime_objects:
        raise BuilderError("compiler input closure lacks the sealed compiler runtime closure")
    static_paths: list[tuple[str, Path]] = []
    for item in runtime_objects:
        record = item.get("record") if isinstance(item, Mapping) else None
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise BuilderError("compiler input closure runtime record is malformed")
        static_paths.append(("compiler-runtime", Path(str(record["path"]))))
    try:
        resource_probe = _sealed_compiler_probe(
            compiler,
            exec_helper,
            compiler_environment,
            Path(str(next(iter(recipes.values()))["cwd"])),
            ["--print-resource-dir"],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuilderError("compiler resource directory discovery could not complete") from exc
    if resource_probe.returncode != 0 or resource_probe.stderr or len(resource_probe.stdout) > COMPILER_BROKER_MAX_OUTPUT:
        raise BuilderError("compiler resource directory discovery failed or exceeded its bounded output")
    resource_dir = _compiler_resource_directory(resource_probe.stdout)
    for path in sorted(resource_dir.rglob("*")):
        if path.is_symlink():
            raise BuilderError("compiler resource directory contains a symlink")
        if path.is_file():
            static_paths.append(("compiler-resource", path))
    bitcode_root = EXPECTED_ROCM_ROOT.resolve(strict=True) / "amdgcn" / "bitcode"
    if bitcode_root.is_symlink() or not bitcode_root.is_dir():
        raise BuilderError("compiler input closure device bitcode root is unavailable")
    for path in sorted(bitcode_root.rglob("*")):
        if path.is_symlink():
            raise BuilderError("compiler input closure device bitcode contains a symlink")
        if path.is_file():
            static_paths.append(("compiler-device-bitcode", path))
    if not any(role == "compiler-device-bitcode" for role, _path in static_paths):
        raise BuilderError("compiler input closure device bitcode set is empty")
    result: dict[str, list[dict[str, str]]] = {}
    for recipe_key, recipe in recipes.items():
        argv = recipe.get("argv")
        cwd = Path(str(recipe.get("cwd")))
        if not isinstance(argv, list) or not cwd.is_absolute():
            raise BuilderError("compiler input closure recipe is malformed")
        scan_argv: list[str] = []
        index = 0
        while index < len(argv):
            value = argv[index]
            if value == "-c":
                index += 1
                continue
            if value == "-o":
                index += 2
                continue
            scan_argv.append(value)
            index += 1
        try:
            scanned = _sealed_compiler_probe(
                compiler,
                exec_helper,
                compiler_environment,
                cwd,
                [*scan_argv, "-M"],
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BuilderError("compiler input dependency discovery could not complete") from exc
        if scanned.returncode != 0 or len(scanned.stdout) > COMPILER_BROKER_MAX_OUTPUT or len(scanned.stderr) > COMPILER_BROKER_MAX_OUTPUT:
            raise BuilderError("compiler input dependency discovery failed or exceeded its bounded output")
        paths = [("compiler-header", path) for path in _dependency_paths_from_make_output(scanned.stdout, label=str(recipe_key))]
        paths.extend(static_paths)
        existing = {str(Path(str(item["path"])).resolve(strict=True)) for item in recipe.get("inputs", ())}
        records: list[dict[str, str]] = []
        seen: set[str] = set()
        for role, path in paths:
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise BuilderError("compiler input closure path disappeared before issuance") from exc
            if str(resolved) in existing or str(resolved) in seen:
                continue
            exact_actions.file_record(path, role=role, label="compiler input closure")
            records.append({"role": role, "path": str(path)})
            seen.add(str(resolved))
        if not records:
            raise BuilderError("compiler input closure contains no transitive compiler inputs")
        result[str(recipe_key)] = records
    return result


def _validate_toolchain() -> None:
    compiler = EXPECTED_ROCM_ROOT / "bin/amdclang++"
    if not EXPECTED_ROCM_ROOT.is_dir() or not compiler.exists() or not os.access(compiler, os.X_OK):
        raise BuilderError("ROCm amdclang++ is unavailable at the pinned logical path")
    contracts.canonical_build_tools()
    # Do not execute or post-hash this pathname here.  build.rs snapshots the
    # verified compiler bytes into a sealed descriptor before its version probe
    # and every CMake use; that is the sole compiler execution authority.


def _materialize_reviewed_snapshot(repo: Path, candidate: Mapping[str, Any], parent: Path) -> Path:
    """Materialize only the reviewed Git object into a private build root."""

    if not parent.is_absolute() or parent.resolve() == repo:
        raise BuilderError("private reviewed build snapshot must remain outside the checkout")
    archive = contracts._git_output_bytes(repo, ("archive", "--format=tar", str(candidate["reviewed_sha"])))
    snapshot = Path(tempfile.mkdtemp(prefix="sllm-rmsnorm-g1-reviewed-", dir=str(parent)))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
            for member in stream:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise BuilderError("reviewed Git snapshot contains an unsafe archive member")
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    destination.mkdir(exist_ok=True)
                    continue
                source = stream.extractfile(member)
                if source is None:
                    raise BuilderError("reviewed Git snapshot file is unreadable")
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o500)
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as output:
                        shutil.copyfileobj(source, output)
                finally:
                    os.close(descriptor)
                destination.chmod((member.mode & 0o555) or 0o400)
        return snapshot
    except BaseException:
        raise


def _binary(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1 or not os.access(path, os.X_OK):
        raise BuilderError(f"{label} is not a non-empty regular executable")


def _compiler_execution(source: Mapping[str, Any], broker: CompilerBroker) -> dict[str, Any]:
    """Materialize a transcript only from the closed broker instance."""

    if not isinstance(broker, CompilerBroker) or broker.source != dict(source) or not broker._closing:
        raise BuilderError("static compiler JSON is not a closed parent-owned broker transcript")
    return broker.transcript()


def _extract_companion(binary: Path, target: str, output: Path, *, cwd: Path, env: Mapping[str, str]) -> None:
    tools = {
        "objcopy": EXPECTED_ROCM_ROOT / "lib/llvm/bin/llvm-objcopy",
        "bundler": EXPECTED_ROCM_ROOT / "lib/llvm/bin/clang-offload-bundler",
    }
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()):
        raise BuilderError("pinned code-object inspection tools are unavailable")
    with tempfile.TemporaryDirectory(prefix="sllm-semantic-g1-extract-", dir="/tmp") as temporary:
        root = Path(temporary)
        fatbin, copy_output, extracted = root / "embedded.hip_fatbin", root / "copy-output", root / "device.elf"
        _run([str(tools["objcopy"]), f"--dump-section=.hip_fatbin={fatbin}", str(binary), str(copy_output)], cwd=cwd, env=env, timeout=120.0)
        if not fatbin.is_file() or fatbin.is_symlink() or fatbin.stat().st_size < 1:
            raise BuilderError("semantic runtime has no extractable HIP fatbin")
        listed = _run([str(tools["bundler"]), "--list", "--type=o", f"--input={fatbin}"], cwd=cwd, env=env, timeout=120.0)
        bundles = [line.strip() for line in listed.stdout.decode("utf-8", "strict").splitlines() if line.strip()]
        expected = [f"hipv4-amdgcn-amd-amdhsa--{target}", "host-x86_64-unknown-linux-gnu-"]
        if bundles != expected:
            raise BuilderError("semantic runtime fatbin does not contain exactly the target and host bundle")
        _run([str(tools["bundler"]), "--unbundle", "--type=o", f"--targets={expected[0]}", f"--input={fatbin}", f"--output={extracted}"], cwd=cwd, env=env, timeout=120.0)
        if extracted.is_symlink() or not extracted.is_file() or extracted.stat().st_size < 1:
            raise BuilderError("cannot extract the target-qualified device companion")
        shutil.copyfile(extracted, output, follow_symlinks=False)
        os.chmod(output, 0o600)


def _loader_record(binary: Path) -> dict[str, Any]:
    descriptor = contracts._open_regular(binary, "staged semantic runtime")
    try:
        interpreter = contracts.elf_interpreter_path(descriptor)
    finally:
        os.close(descriptor)
    if interpreter is None:
        raise BuilderError("semantic runtime has no dynamic loader")
    # Keep the literal PT_INTERP path in path while retaining its canonical
    # target in resolved_path. The controller can then match sealed bytes
    # without resolving a mutable pathname again.
    resolved = interpreter.resolve(strict=True)
    descriptor = contracts._open_regular(resolved, "runtime loader")
    try:
        return contracts._record_from_descriptor(descriptor, path=interpreter)
    finally:
        os.close(descriptor)


def _runtime_dependency_records(binary: Path, loader_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        closure = contracts.runtime_dependency_closure(binary)
    except (contracts.EvidenceError, OSError) as exc:
        raise BuilderError("complete recursive PT_INTERP/DT_NEEDED runtime closure could not be captured") from exc
    objects = closure.get("objects")
    if not isinstance(objects, list) or len(objects) < 3:
        raise BuilderError("runtime dependency closure does not contain executable, loader, and dependencies")
    root = str(binary.resolve(strict=True))
    libraries: list[dict[str, Any]] = []
    for item in objects:
        record = item.get("record") if isinstance(item, Mapping) else None
        if not isinstance(record, Mapping):
            raise BuilderError("runtime dependency closure object record is malformed")
        if str(record.get("resolved_path")) in {root, loader_path}:
            continue
        libraries.append({"name": Path(str(record["resolved_path"])).name, "record": dict(record)})
    if not libraries:
        raise BuilderError("runtime dependency closure has no dynamic dependencies")
    return closure, libraries


def _materialize_exec_helper(path: Path, *, cwd: Path, authority: Mapping[str, Any]) -> None:
    """Build and freeze the only native code allowed between spawn and exec."""

    if path.exists() or path.is_symlink():
        raise BuilderError("compiler exec helper output already exists")
    cxx = contracts.CANONICAL_CXX
    expected = authority.get("toolchain", {}).get("cxx") if isinstance(authority.get("toolchain"), Mapping) else None
    if not isinstance(expected, Mapping) or contracts.file_identity(cxx, "compiler exec helper C++ tool") != expected:
        raise BuilderError("compiler exec helper C++ tool drifted from the reviewed authority")
    try:
        completed = subprocess.run(
            [str(cxx), "-x", "c++", "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti", "-o", str(path), "-"],
            cwd=cwd,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            input=COMPILER_EXEC_HELPER_SOURCE.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuilderError("compiler exec helper could not be compiled") from exc
    if completed.returncode != 0:
        raise BuilderError(
            "compiler exec helper compilation failed: "
            + completed.stderr.decode("utf-8", "replace")[-1000:]
        )
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise BuilderError("compiler exec helper output is not a regular file")
    os.chmod(path, 0o555)


def build_runtime_artifact(*, repo: Path = ROOT, row_id: str, identity: Mapping[str, Any], authority: Mapping[str, Any], output_dir: Path, timeout_seconds: float = MAX_BUILD_TIMEOUT_SECONDS) -> BuildResult:
    if not COMPILER_BROKER_AVAILABLE:
        raise BuilderError("semantic G1 compiler broker is unavailable")
    if row_id not in contracts.ROWS or timeout_seconds <= 0 or timeout_seconds > MAX_BUILD_TIMEOUT_SECONDS:
        raise BuilderError("semantic G1 build row or timeout is invalid")
    repo = contracts.canonical_repository(repo)
    candidate = contracts.verify_repository_identity(repo, identity)
    if contracts._validate_authority_document(authority, candidate) != dict(authority):
        raise BuilderError("builder authority is not the reviewed controller authority")
    for name, path in contracts.CANONICAL_TOOL_PATHS.items():
        if contracts.file_identity(path, f"semantic G1 toolchain {name}") != authority["toolchain"][name]:
            raise BuilderError(f"semantic G1 toolchain executable {name} drifted from authority")
    matrix = contracts.validate_matrix(repo)
    row = contracts.row_by_id(matrix, row_id)
    _validate_inherited_environment(row["target"])
    _validate_toolchain()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute() or output_dir.name != row_id:
        raise BuilderError("builder output must be an absolute target-qualified row directory")
    _private_directory(output_dir.parent, "builder output root")
    _new_directory(output_dir, "builder row output")
    cargo_target_dir = output_dir.parent / f"cargo-target-{row['target']}"
    _new_directory(cargo_target_dir, "builder target directory")
    native_build_dir = output_dir.parent / f"native-hip-build-{row['target']}"
    _new_directory(native_build_dir, "parent-derived native HIP build directory")
    build_repo = _materialize_reviewed_snapshot(repo, candidate, output_dir.parent)
    command = list(contracts.EXPECTED_COMMAND)
    compiler_snapshot = contracts.snapshot_file(
        EXPECTED_ROCM_ROOT / "bin/amdclang++", contracts.COMPILER_SOURCE_RECORD,
        "reviewed pre-build ROCm compiler",
    )
    compiler_source = dict(compiler_snapshot.record)
    exec_helper = build_repo / "ci/tools" / COMPILER_EXEC_HELPER_NAME
    _materialize_exec_helper(exec_helper, cwd=build_repo, authority=authority)
    client_path = build_repo / "ci/tools/compiler-client.py"
    _write_new(client_path, COMPILER_CLIENT_TEMPLATE.encode("utf-8"), "compiler broker client", mode=0o500)
    for directory, directories, files in os.walk(build_repo):
        del files
        Path(directory).chmod(0o555)
        for name in directories:
            (Path(directory) / name).chmod(0o555)
    client_path.chmod(0o555)
    environment = build_environment(row["target"], cargo_target_dir, native_build_dir)
    exact_compiler_environment = compiler_spawn_environment(environment, str(compiler_source["path"]))
    action_recipes = _semantic_exact_action_recipes(build_repo, native_build_dir, row["target"])
    for recipe_key, closure_inputs in _compiler_input_closures(
        compiler_snapshot, exec_helper, exact_compiler_environment, action_recipes
    ).items():
        action_recipes[recipe_key]["inputs"].extend(closure_inputs)
        action_recipes[recipe_key]["implicit"].append({
            "role": "compiler-input-closure-policy",
            "bytes": exact_actions.canonical_bytes({
                "algorithm": "sealed-amdclang-resource-preprocess-device-runtime-closure-v2",
                "live_pre_exec_validation": True,
                "final_environment_sha256": exact_actions.sha256(exact_compiler_environment),
            }),
        })
    broker = CompilerBroker(
        compiler=compiler_snapshot,
        client_path=client_path,
        exec_helper=exec_helper,
        socket_root=native_build_dir,
        allowed_roots=(build_repo, cargo_target_dir, native_build_dir),
        output_roots=(cargo_target_dir, native_build_dir),
        reviewed_sources={str(item["path"]): item for item in authority["sources"]},
        reviewed_tools=authority["toolchain"],
        target=row["target"],
        expected_environment=environment,
        compiler_environment=exact_compiler_environment,
        action_recipes=action_recipes,
        require_complete_recipe_set=True,
    )
    broker.start()
    try:
        _run(
            command, cwd=build_repo, env=environment, timeout=timeout_seconds,
            compiler_source=compiler_source, broker=broker,
        )
        compiler_execution = _compiler_execution(compiler_source, broker)
    finally:
        if not broker._closing:
            broker.abort()
        compiler_snapshot.close()
    source = cargo_target_dir / "release" / contracts.BINARY_NAME
    _binary(source, "exact row Cargo output")
    staged = output_dir / contracts.BINARY_NAME
    shutil.copyfile(source, staged, follow_symlinks=False)
    os.chmod(staged, 0o700)
    _binary(staged, "staged semantic runtime")
    companion = output_dir / contracts.COMPANION_NAME.format(target=row["target"])
    _extract_companion(staged, row["target"], companion, cwd=build_repo, env=environment)
    if companion.is_symlink() or not companion.is_file() or companion.stat().st_size < 1:
        raise BuilderError("device companion is missing after extraction")
    loader = _loader_record(staged)
    closure, libraries = _runtime_dependency_records(staged, str(loader["resolved_path"]))
    records = {
        "binary": contracts.file_identity(staged, "staged semantic runtime"),
        "companion": contracts.file_identity(companion, "device companion"),
        "loader": loader,
    }
    metadata = {
        "schema_version": "rmsnorm-semantic-g1-artifact-v1",
        "metadata_id": f"rmsnorm-semantic-g1-artifact-{row['target']}",
        "row_id": row["row_id"],
        "target": row["target"],
        "candidate": candidate,
        "authority": dict(authority),
        "artifact_kind": "rmsnorm-semantic-g1-runtime",
        "command": command,
        "scope": contracts.EXPECTED_SCOPE,
        "codegen": contracts.EXPECTED_CODEGEN[row["target"]],
        "contracts": contracts.authority_contract_hashes(authority),
        "records": records,
        "runtime_libraries": libraries,
        "runtime_dependency_closure": closure,
        "compiler_execution": compiler_execution,
    }
    metadata_path = output_dir / contracts.METADATA_NAME
    _write_new(metadata_path, contracts.canonical_bytes(metadata), "semantic G1 metadata")
    for path, label in ((staged, "runtime binary"), (companion, "device companion"), (metadata_path, "metadata")):
        _sidecar(path, label)
    contracts._validate_metadata(metadata, row=row, identity=identity, repo=repo, authority=authority)
    expected_names = {
        contracts.BINARY_NAME,
        contracts.BINARY_NAME + contracts.SIDECAR_SUFFIX,
        contracts.COMPANION_NAME.format(target=row["target"]),
        contracts.COMPANION_NAME.format(target=row["target"]) + contracts.SIDECAR_SUFFIX,
        contracts.METADATA_NAME,
        contracts.METADATA_NAME + contracts.SIDECAR_SUFFIX,
    }
    if {path.name for path in output_dir.iterdir()} != expected_names:
        raise BuilderError("builder row output has stale or unknown files")
    return BuildResult(row["row_id"], row["target"], output_dir, cargo_target_dir, staged, companion, metadata_path, contracts.sha256_file(staged), contracts.sha256_file(companion), contracts.sha256_file(metadata_path), tuple(command), compiler_execution, True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--row", choices=contracts.ROWS, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--timeout-seconds", type=float, default=MAX_BUILD_TIMEOUT_SECONDS)
    result.add_argument("--reviewed-sha", required=True)
    result.add_argument("--tested-sha", required=True)
    result.add_argument("--workflow-sha", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        identity = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha}
        repo = contracts.controller_workspace(Path(os.environ["GITHUB_WORKSPACE"]))
        authority = contracts.reviewed_authority(repo, identity)
        build_runtime_artifact(repo=repo, row_id=args.row, identity=identity, authority=authority, output_dir=args.output_dir, timeout_seconds=args.timeout_seconds)
    except (BuilderError, ContractError, OSError, TypeError, ValueError) as exc:
        print(f"semantic RMSNorm G1 builder: FAIL: {exc}", file=sys.stderr)
        return 1
    print("semantic RMSNorm G1 builder: staged only; GPU evidence remains controller-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
