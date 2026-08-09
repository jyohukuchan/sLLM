#!/usr/bin/env python3
"""Fixed, stdlib-only worker for controller-owned semantic RMSNorm G1 evidence.

The controller supplies only sealed runtime descriptors and one authenticated
``SOCK_SEQPACKET`` channel.  This module deliberately imports no project code:
when it is executed from its reviewed sealed source with ``/usr/bin/python3
-I``, the worker has no Python-level route to a mutable checkout, a report
writer, or a PASS-emission API.
"""

from __future__ import annotations

import argparse
import array
import base64
import ctypes
import fcntl
import json
import math
import os
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_MAGIC = b"SLLMG1IN"
OUTPUT_MAGIC = b"SLLMG1OT"
INPUT_PROTOCOL_VERSION = 1
OUTPUT_PROTOCOL_VERSION = 2
INPUT_HEADER_BYTES = 112
OUTPUT_HEADER_BYTES = 428
BF16_BYTES = 2
MAX_N = 4096
MAX_ELEMENTS = 262144
MAX_OUTPUT = 1024 * 1024
MAX_RUNTIME_RSS_BYTES = 2 * 1024 * 1024 * 1024
PROCESS_LIMITER = "/usr/bin/prlimit"
PROCESS_ADDRESS_LIMIT_BYTES = 64 * 1024 * 1024 * 1024
PROCESS_COUNT_LIMIT = 4096
TERM_GRACE_SECONDS = 0.25
KILL_GRACE_SECONDS = 1.0
KERNEL_SYMBOL = "rmsnorm.baseline.wave32.v1"
DEVICE_SYMBOL = "sllm_rmsnorm_baseline_wave32_v1"
TARGETS = ("gfx1030", "gfx1201")
ROWS = tuple(f"rmsnorm-semantic-g1-{target}" for target in TARGETS)
EXPECTED_CASE_COUNT = 15
SANITIZED_RUNTIME_PATH = "/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SANITIZED_RUNTIME_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib"
_REQUIRED_SEALS = (
    getattr(fcntl, "F_SEAL_SHRINK", 0)
    | getattr(fcntl, "F_SEAL_GROW", 0)
    | getattr(fcntl, "F_SEAL_WRITE", 0)
    | getattr(fcntl, "F_SEAL_SEAL", 0)
)
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


class RunnerError(ValueError):
    """A malformed raw frame or a failed Linux containment boundary."""


@dataclass(frozen=True)
class RawExecution:
    exit_code: int | None
    timed_out: bool
    crashed: bool
    cleanup_proven: bool
    stdout: bytes
    stderr: bytes
    error: str | None


@dataclass(frozen=True)
class _ProcFacts:
    starttime: int
    parent_pid: int
    pgrp: int
    rss_bytes: int


def _proc_facts(pid: int) -> _ProcFacts | None:
    """Read one Linux process identity and RSS, rejecting malformed proc data."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(") ", 1)[1].split()
        rss_pages = int(fields[21])
        page_size = os.sysconf("SC_PAGE_SIZE")
        if rss_pages < 0 or page_size <= 0:
            return None
        return _ProcFacts(int(fields[19]), int(fields[1]), int(fields[2]), rss_pages * page_size)
    except (OSError, IndexError, ValueError):
        return None


def _proc_snapshot() -> dict[int, _ProcFacts]:
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise RunnerError("Linux /proc containment support is unavailable") from exc
    snapshot: dict[int, _ProcFacts] = {}
    for entry in entries:
        if entry.name.isdecimal():
            facts = _proc_facts(int(entry.name))
            if facts is not None:
                snapshot[int(entry.name)] = facts
    return snapshot


def _child_subreaper_state() -> bool:
    """Read the process-local subreaper state through the Linux primitive."""

    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RunnerError("semantic G1 requires Linux pidfd and child-subreaper containment")
    value = ctypes.c_int()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(value), 0, 0, 0)
    except OSError as exc:
        raise RunnerError("cannot inspect Linux child-subreaper containment") from exc
    if result != 0:
        raise RunnerError("Linux child-subreaper containment state is unavailable")
    return value.value != 0


def _set_child_subreaper(enabled: bool) -> None:
    """Set the process-local reaper flag, rejecting a missing Linux primitive."""

    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RunnerError("semantic G1 requires Linux pidfd and child-subreaper containment")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0)
    except OSError as exc:
        raise RunnerError("cannot configure Linux child-subreaper containment") from exc
    if result != 0:
        raise RunnerError("Linux child-subreaper containment was refused")


def _enable_child_subreaper() -> bool:
    """Enable reaping and return the caller's prior process-local state."""

    prior = _child_subreaper_state()
    _set_child_subreaper(True)
    return prior


@dataclass
class LinuxContainment:
    """PID/starttime-bound containment for a process and escaped descendants.

    A child can call ``setsid`` and close every pipe, which makes ordinary
    process-group cleanup insufficient.  The parent becomes a Linux child
    subreaper before launch, tracks PID/starttime identities from ``/proc``,
    signals those identities through pidfds, and reaps adopted descendants.
    A missing primitive or a surviving identity is a hard failure.
    """

    baseline: dict[int, _ProcFacts]
    parent_pid: int
    prior_subreaper: bool
    tracked: dict[int, _ProcFacts] = field(default_factory=dict)
    root_pid: int | None = None
    root_pgrp: int | None = None
    subreaper_restored: bool = False
    quiescence_rounds: int = 0

    @classmethod
    def begin(cls) -> "LinuxContainment":
        prior_subreaper = _enable_child_subreaper()
        try:
            return cls(_proc_snapshot(), os.getpid(), prior_subreaper)
        except BaseException:
            if not prior_subreaper:
                _set_child_subreaper(False)
            raise

    def bind_root(self, pid: int, pgrp: int) -> None:
        facts = _proc_facts(pid)
        if facts is None or pgrp < 1:
            raise RunnerError("cannot bind a live contained process root")
        self.root_pid, self.root_pgrp = pid, pgrp
        self.tracked[pid] = facts
        self.observe()

    def observe(self) -> None:
        if self.root_pid is None or self.root_pgrp is None:
            raise RunnerError("contained process root is not bound")
        facts = _proc_snapshot()
        related = set(self.tracked)
        changed = True
        while changed:
            changed = False
            for pid, observed in facts.items():
                if pid in related:
                    continue
                # A member of the launch process group is contained directly.
                # A normal descendant is detected through its parent.  Once a
                # double-fork/setns escape is adopted by this subreaper it is a
                # new direct child; baseline identities prevent unrelated
                # pre-existing children from being swept into this operation.
                baseline = self.baseline.get(pid)
                is_new = baseline is None or baseline.starttime != observed.starttime
                if (
                    observed.pgrp == self.root_pgrp
                    or observed.parent_pid in related
                    or (is_new and observed.parent_pid == self.parent_pid)
                ):
                    related.add(pid)
                    self.tracked[pid] = observed
                    changed = True

    def alive(self) -> dict[int, _ProcFacts]:
        alive: dict[int, _ProcFacts] = {}
        for pid, expected in self.tracked.items():
            observed = _proc_facts(pid)
            if observed is not None and observed.starttime == expected.starttime:
                alive[pid] = observed
        return alive

    def rss_bytes(self) -> int:
        self.observe()
        return sum(facts.rss_bytes for facts in self.alive().values())

    def assert_rss_within(self, limit: int) -> None:
        if limit < 1:
            raise RunnerError("contained RSS limit is invalid")
        if self.rss_bytes() > limit:
            raise RunnerError("contained process tree exceeded its RSS limit")

    @staticmethod
    def _pidfd_signal(pid: int, expected: _ProcFacts, signal_value: signal.Signals) -> bool:
        observed = _proc_facts(pid)
        if observed is None:
            return True
        if observed.starttime != expected.starttime:
            return False
        try:
            descriptor = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        try:
            # Recheck after obtaining the pidfd: the pidfd is the actual
            # anti-replay identity used for the signal operation.
            observed = _proc_facts(pid)
            if observed is None:
                return True
            if observed.starttime != expected.starttime:
                return False
            signal.pidfd_send_signal(descriptor, signal_value, None, 0)
            return True
        except ProcessLookupError:
            return True
        except OSError:
            return False
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _reap(self, expected: Mapping[int, _ProcFacts]) -> bool:
        for pid, identity in expected.items():
            observed = _proc_facts(pid)
            if observed is not None and observed.starttime != identity.starttime:
                return False
            try:
                waited, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # A direct group member may still be parented by the direct
                # child.  After that child exits the subreaper owns it; this is
                # checked again in the outer cleanup loop.
                continue
            except OSError:
                return False
            if waited not in (0, pid):
                return False
        return True

    def _restore_subreaper_after_reap(self) -> bool:
        """Undo only our temporary reaper flag after there are no survivors.

        Leaving the flag enabled after a normal host invocation changes how
        later unrelated subprocess trees are adopted.  Restoring it before
        all PID/starttime-bound descendants are dead would weaken containment,
        so a restore failure (or a survivor) remains fail-closed.
        """

        if self.subreaper_restored:
            return True
        if self.prior_subreaper:
            self.subreaper_restored = True
            return True
        if self.alive():
            return False
        try:
            _set_child_subreaper(False)
            if _child_subreaper_state():
                return False
        except RunnerError:
            return False
        self.subreaper_restored = True
        return True

    def terminate_and_reap(self, process: subprocess.Popen[bytes]) -> bool:
        if self.root_pid is None or self.root_pgrp is None:
            return False
        for signal_value, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, KILL_GRACE_SECONDS)):
            try:
                self.observe()
            except RunnerError:
                return False
            alive = self.alive()
            if not alive:
                break
            # Process-group delivery handles normal descendants cheaply.  A
            # missing group is benign only after every PID/starttime identity
            # below has also been inspected and pidfd-signalled.
            try:
                os.killpg(self.root_pgrp, signal_value)
            except ProcessLookupError:
                pass
            except OSError:
                return False
            for pid, identity in list(alive.items()):
                if not self._pidfd_signal(pid, identity, signal_value):
                    return False
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                try:
                    self.observe()
                except RunnerError:
                    return False
                alive = self.alive()
                if not alive:
                    break
                if not self._reap(alive):
                    return False
                time.sleep(0.01)
        try:
            process.wait(timeout=KILL_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            return False
        # A single /proc scan is not a closure proof: an adopted descendant
        # may be between fork/setns/setsid and the scan.  Require repeated
        # stable observations after the root has exited, with no new tracked
        # identity and no process-group member left alive.
        stable = 0
        previous_tracked = set(self.tracked)
        deadline = time.monotonic() + KILL_GRACE_SECONDS
        while time.monotonic() < deadline and stable < 3:
            try:
                self.observe()
            except RunnerError:
                return False
            alive = self.alive()
            if alive:
                for pid, identity in list(alive.items()):
                    if not self._pidfd_signal(pid, identity, signal.SIGKILL):
                        return False
                if not self._reap(alive):
                    return False
                stable = 0
                previous_tracked = set(self.tracked)
                time.sleep(0.01)
                continue
            current_tracked = set(self.tracked)
            group_members = _group_members(self.root_pgrp)
            if group_members or current_tracked != previous_tracked:
                stable = 0
                previous_tracked = current_tracked
            else:
                stable += 1
            if not self._reap(self.tracked):
                return False
            time.sleep(0.02)
        if stable < 3:
            self.quiescence_rounds = stable
            return False
        self.quiescence_rounds = stable
        try:
            self.observe()
        except RunnerError:
            return False
        if self.alive() or _group_members(self.root_pgrp):
            return False
        return self._reap(self.tracked) and self._restore_subreaper_after_reap()

    def restore_after_launch_failure(self) -> bool:
        """Restore the temporary reaper even when Popen never returned."""

        if self.tracked or self.alive():
            return False
        return self._restore_subreaper_after_reap()


def _bf16_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def _f32_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return ((bits + (((bits >> 16) & 1) + 0x7FFF)) >> 16) & 0xFFFF


def _f32(value: Any) -> float:
    """Round through IEEE-754 binary32 at every specified oracle boundary."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def independent_rmsnorm_oracle(activation: bytes, raw_scale: bytes, rows: int, width: int, epsilon: float) -> bytes:
    if rows < 1 or width < 1 or width > MAX_N or len(activation) != rows * width * BF16_BYTES or len(raw_scale) != width * BF16_BYTES:
        raise RunnerError("independent RMSNorm oracle payload is out of contract")
    scales = [_f32(_bf16_to_f32(item[0])) for item in struct.iter_unpack("<H", raw_scale)]
    result: list[int] = []
    for row in range(rows):
        values = [_f32(_bf16_to_f32(item[0])) for item in struct.iter_unpack("<H", activation[row * width * 2:(row + 1) * width * 2])]
        sum_squares = _f32(0.0)
        for value in values:
            sum_squares = _f32(sum_squares + _f32(value * value))
        mean_square = _f32(sum_squares / _f32(width))
        inverse = _f32(_f32(1.0) / _f32(math.sqrt(_f32(mean_square + _f32(epsilon)))))
        for index, value in enumerate(values):
            result.append(_f32_to_bf16(_f32(_f32(value * inverse) * _f32(_f32(1.0) + scales[index]))))
    return b"".join(struct.pack("<H", value) for value in result)


def encode_request(shape: tuple[int, ...], epsilon: float, activation: bytes, raw_scale: bytes) -> bytes:
    if not 1 <= len(shape) <= 8 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape):
        raise RunnerError("runtime request shape is invalid")
    elements = math.prod(shape)
    if shape[-1] > MAX_N or elements > MAX_ELEMENTS or len(activation) != elements * 2 or len(raw_scale) != shape[-1] * 2:
        raise RunnerError("runtime request payload exceeds the semantic G1 bound")
    return b"".join((
        INPUT_MAGIC,
        struct.pack("<II", INPUT_PROTOCOL_VERSION, INPUT_HEADER_BYTES),
        struct.pack("<II", len(shape), 0),
        struct.pack("<8Q", *(list(shape) + [0] * (8 - len(shape)))),
        struct.pack("<IIQQ", struct.unpack("<I", struct.pack("<f", epsilon))[0], 0, len(activation), len(raw_scale)),
        activation,
        raw_scale,
    ))


def _fixed_string(data: bytes, offset: int) -> tuple[str, int]:
    raw = data[offset:offset + 64]
    if len(raw) != 64 or b"\0" not in raw:
        raise RunnerError("runtime protocol fixed string is malformed")
    try:
        return raw.split(b"\0", 1)[0].decode("ascii"), offset + 64
    except UnicodeDecodeError as exc:
        raise RunnerError("runtime protocol fixed string is not ASCII") from exc


def parse_response(data: bytes, *, expected_target: str, expected_device_index: int, expected_shape: tuple[int, ...], expected_epsilon: float) -> dict[str, Any]:
    """Parse raw v2 bytes and bind every dispatch/resource field to G1."""

    if len(data) < OUTPUT_HEADER_BYTES or data[:8] != OUTPUT_MAGIC:
        raise RunnerError("runtime response magic or minimum size is invalid")
    offset = 8

    def u32() -> int:
        nonlocal offset
        value = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        return value

    def u64() -> int:
        nonlocal offset
        value = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return value

    version, header_bytes = u32(), u32()
    if version != OUTPUT_PROTOCOL_VERSION or header_bytes != OUTPUT_HEADER_BYTES:
        raise RunnerError("runtime protocol lacks v2 raw resource evidence")
    rank, reserved = u32(), u32()
    shape = struct.unpack_from("<8Q", data, offset)
    offset += 64
    element_count, normalized_size, row_count = u64(), u64(), u64()
    epsilon_bits, epsilon_reserved = u32(), u32()
    device_index, backend = u32(), u32()
    dispatch_id, dispatch_count, kernel_id, workgroup, grid = u64(), u32(), u32(), u32(), u32()
    fallback_allowed, fallback_used = u32(), u32()
    semantic_wire, contract_version, accumulation_wire, input_count, output_count, binding_count, extension_reserved = (u32() for _ in range(7))
    allocation_count, copy_count, kernel_count, resource_reserved = u32(), u32(), u32(), u32()
    cleanup_pending, cleanup_durable, cleanup_errors = u64(), u64(), u64()
    kernel_symbol, offset = _fixed_string(data, offset)
    device_symbol, offset = _fixed_string(data, offset)
    target, offset = _fixed_string(data, offset)
    output_len = u64()
    expected_elements = math.prod(expected_shape)
    expected_rows = math.prod(expected_shape[:-1])
    epsilon_bits_expected = struct.unpack("<I", struct.pack("<f", expected_epsilon))[0]
    if (
        not 1 <= rank <= 8 or tuple(shape[:rank]) != expected_shape or any(shape[rank:])
        or any((reserved, epsilon_reserved, extension_reserved, resource_reserved))
        or (element_count, normalized_size, row_count) != (expected_elements, expected_shape[-1], expected_rows)
        or epsilon_bits != epsilon_bits_expected
    ):
        raise RunnerError("runtime response shape/count/epsilon contract is invalid")
    if (
        target != expected_target or device_index != expected_device_index or backend != 1
        or dispatch_id < 1 or dispatch_count != 1 or kernel_id != 1 or workgroup != 256 or grid != expected_rows
        or fallback_allowed != 0 or fallback_used != 0 or semantic_wire != 1 or contract_version != 1
        or accumulation_wire != 3 or (input_count, output_count, binding_count) != (2, 1, 3)
        or (cleanup_pending, cleanup_durable, cleanup_errors) != (0, 0, 0)
        or kernel_symbol != KERNEL_SYMBOL or device_symbol != DEVICE_SYMBOL
    ):
        raise RunnerError("runtime response dispatch/backend/fallback contract is invalid")
    observed_counts = {
        "allocation_count": allocation_count,
        "copy_count": copy_count,
        "dispatch_count": dispatch_count,
        "kernel_count": kernel_count,
    }
    if observed_counts != {"allocation_count": 3, "copy_count": 3, "dispatch_count": 1, "kernel_count": 1}:
        raise RunnerError("runtime raw resource counts are not the exact semantic-G1 protocol counts")
    if output_len != expected_elements * BF16_BYTES or offset + output_len != len(data):
        raise RunnerError("runtime response output length or trailing bytes are invalid")
    return {
        "target": target, "device_index": device_index, "shape": list(expected_shape),
        "element_count": element_count, "row_count": row_count, "normalized_size": normalized_size,
        "dispatch_id": dispatch_id, "dispatch_count": dispatch_count, "kernel_id": kernel_id,
        "kernel_symbol": kernel_symbol, "device_symbol": device_symbol,
        "resource_counts": observed_counts, "output": data[offset:offset + output_len],
    }


def _group_members(pgid: int) -> list[int]:
    return sorted(pid for pid, facts in _proc_snapshot().items() if facts.pgrp == pgid)


def cleanup_process_group(process: subprocess.Popen[bytes], pgid: int) -> bool:
    """Compatibility entry point with full subreaper/pidfd cleanup proof."""

    try:
        containment = LinuxContainment.begin()
        containment.bind_root(process.pid, pgid)
        return containment.terminate_and_reap(process)
    except RunnerError:
        return False


def close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def bounded_exchange(
    process: subprocess.Popen[bytes], request: bytes, timeout_seconds: float,
    *, containment: LinuxContainment | None = None, rss_limit_bytes: int = MAX_RUNTIME_RSS_BYTES,
) -> tuple[bytes, bytes]:
    """Bound IO, wall time, output, and full contained-tree RSS."""

    if process.stdin is None or process.stdout is None or process.stderr is None or timeout_seconds <= 0:
        raise RunnerError("runtime child pipes or timeout are invalid")
    stdin_fd, stdout_fd, stderr_fd = process.stdin.fileno(), process.stdout.fileno(), process.stderr.fileno()
    for descriptor in (stdin_fd, stdout_fd, stderr_fd):
        os.set_blocking(descriptor, False)
    buffers = {stdout_fd: bytearray(), stderr_fd: bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(stdin_fd, selectors.EVENT_WRITE, "stdin")
    selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
    selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
    position = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            if containment is not None:
                containment.assert_rss_within(rss_limit_bytes)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("semantic G1 runtime", timeout_seconds)
            events = selector.select(min(remaining, 0.05))
            if not events:
                continue
            for key, mask in events:
                descriptor = int(key.fileobj)
                if key.data == "stdin" and mask & selectors.EVENT_WRITE:
                    if position < len(request):
                        try:
                            position += os.write(descriptor, request[position:position + 65536])
                        except BlockingIOError:
                            continue
                    if position == len(request):
                        selector.unregister(descriptor)
                        try:
                            process.stdin.close()
                        except OSError:
                            pass
                elif key.data in {"stdout", "stderr"} and mask & selectors.EVENT_READ:
                    try:
                        chunk = os.read(descriptor, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(descriptor)
                        continue
                    buffers[descriptor].extend(chunk)
                    if len(buffers[descriptor]) > MAX_OUTPUT:
                        raise RunnerError("runtime child output exceeded its bounded limit")
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
        if containment is not None:
            containment.assert_rss_within(rss_limit_bytes)
        return bytes(buffers[stdout_fd]), bytes(buffers[stderr_fd])
    finally:
        selector.close()


def _identity_tuple(descriptor: int) -> tuple[int, int]:
    details = os.fstat(descriptor)
    return details.st_dev, details.st_ino


def _parent_open_identities() -> dict[int, tuple[int, int]]:
    identities: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc/self/fd").iterdir():
        if entry.name.isdigit():
            try:
                identities[int(entry.name)] = _identity_tuple(int(entry.name))
            except OSError:
                pass
    return identities


def _process_real_ids(pid: int) -> tuple[int, int]:
    try:
        values: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith(("Uid:", "Gid:")):
                name, raw = line.split(":", 1)
                values[name] = int(raw.split()[0])
        return values["Uid"], values["Gid"]
    except (OSError, KeyError, ValueError, IndexError) as exc:
        raise RunnerError("cannot inspect process credentials") from exc


def process_binding(pid: int) -> dict[str, int]:
    facts = _proc_facts(pid)
    if facts is None:
        raise RunnerError("cannot inspect process binding")
    uid, gid = _process_real_ids(pid)
    return {"pid": pid, "starttime": facts.starttime, "uid": uid, "gid": gid}


def verify_process_binding(binding: Mapping[str, Any]) -> None:
    if set(binding) != {"pid", "starttime", "uid", "gid"} or any(isinstance(value, bool) or not isinstance(value, int) for value in binding.values()):
        raise RunnerError("process binding is not closed")
    if int(binding["pid"]) < 1 or process_binding(int(binding["pid"])) != dict(binding):
        raise RunnerError("process PID/starttime/UID/GID binding changed")


def audit_child_fd_inheritance(
    pid: int,
    expected_descriptors: Sequence[int],
    *,
    expected_binding: Mapping[str, Any],
    optional_descriptors: Sequence[int] = (),
    forbidden_descriptors: Sequence[int] = (),
    parent_identities: Mapping[int, tuple[int, int]] | None = None,
) -> None:
    verify_process_binding(expected_binding)
    if expected_binding.get("pid") != pid:
        raise RunnerError("FD audit PID does not match its bound target child")
    expected = {int(fd): _identity_tuple(int(fd)) for fd in expected_descriptors}
    optional = {_identity_tuple(int(fd)) for fd in optional_descriptors}
    forbidden = {_identity_tuple(int(fd)) for fd in forbidden_descriptors}
    permitted = set(expected.values()) | optional
    if permitted & forbidden:
        raise RunnerError("FD audit permitted and forbidden identities collide")
    inherited = dict(parent_identities or _parent_open_identities())
    inherited_identities = set(inherited.values()) - permitted
    child_root = Path(f"/proc/{pid}/fd")
    try:
        entries = {int(entry.name) for entry in child_root.iterdir() if entry.name.isdigit()}
    except OSError as exc:
        raise RunnerError("FD audit cannot observe a live target child") from exc
    if not set(expected).issubset(entries):
        raise RunnerError("target child closed an expected sealed descriptor before request delivery")
    for descriptor, identity in expected.items():
        try:
            observed = os.stat(child_root / str(descriptor))
        except OSError as exc:
            raise RunnerError("FD audit cannot inspect expected target descriptor") from exc
        if (observed.st_dev, observed.st_ino) != identity:
            raise RunnerError("target child expected descriptor identity changed")
    for descriptor in entries:
        if descriptor <= 2 or descriptor in expected:
            continue
        try:
            observed_stat = os.stat(child_root / str(descriptor))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RunnerError("FD audit cannot inspect target descriptor") from exc
        observed = (observed_stat.st_dev, observed_stat.st_ino)
        if observed in forbidden:
            raise RunnerError("target child inherited an explicitly forbidden controller descriptor")
        if observed not in permitted and observed in inherited_identities:
            raise RunnerError("target child inherited an unrelated parent descriptor")


def descriptor_is_sealed(descriptor: int) -> bool:
    try:
        return (fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _REQUIRED_SEALS) == _REQUIRED_SEALS
    except OSError:
        return False


def fd_path(descriptor: int) -> Path:
    if not isinstance(descriptor, int) or descriptor < 0:
        raise RunnerError("runtime descriptor is invalid")
    return Path(f"/proc/self/fd/{descriptor}")


def gpu_child_pass_fds(loader_fd: int, executable_fd: int, runtime_fds: Sequence[int]) -> tuple[int, ...]:
    values = (loader_fd, executable_fd, *runtime_fds)
    if len(runtime_fds) < 1 or len(set(values)) != len(values) or any(not isinstance(value, int) or value < 0 for value in values):
        raise RunnerError("GPU child retained descriptor set is invalid")
    return values


def semantic_runtime_environment(physical_hip_index: int) -> dict[str, str]:
    if type(physical_hip_index) is not int or physical_hip_index < 0:
        raise RunnerError("semantic G1 physical HIP index is invalid")
    return {
        "PATH": SANITIZED_RUNTIME_PATH,
        "LD_LIBRARY_PATH": SANITIZED_RUNTIME_LD_LIBRARY_PATH,
        "HIP_VISIBLE_DEVICES": str(physical_hip_index),
    }


def execute_raw_runtime(
    *,
    loader_fd: int,
    executable_fd: int,
    runtime_fds: Sequence[int],
    target: str,
    physical_hip_index: int,
    logical_device_index: int,
    request: bytes,
    timeout_seconds: float,
    forbidden_descriptors: Sequence[int] = (),
) -> RawExecution:
    """Launch only controller-sealed bytes and prove no descendant survives."""

    if target not in TARGETS or logical_device_index != 0:
        raise RunnerError("runtime target/logical device is not canonical")
    pass_fds = gpu_child_pass_fds(loader_fd, executable_fd, runtime_fds)
    if any(not descriptor_is_sealed(descriptor) for descriptor in pass_fds):
        raise RunnerError("runtime launch refuses an unsealed descriptor")
    if not os.path.isfile(PROCESS_LIMITER) or not os.access(PROCESS_LIMITER, os.X_OK):
        raise RunnerError("semantic G1 requires the fixed prlimit containment primitive")
    command = [
        PROCESS_LIMITER, f"--as={PROCESS_ADDRESS_LIMIT_BYTES}", f"--nproc={PROCESS_COUNT_LIMIT}", "--",
        str(fd_path(loader_fd)), "--preload", ":".join(str(fd_path(descriptor)) for descriptor in runtime_fds),
        str(fd_path(executable_fd)), "--device-index", str(logical_device_index), "--target", target,
    ]
    process: subprocess.Popen[bytes] | None = None
    containment: LinuxContainment | None = None
    result: RawExecution | None = None
    parent_identities = _parent_open_identities()
    try:
        containment = LinuxContainment.begin()
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/",
            env=semantic_runtime_environment(physical_hip_index), start_new_session=True, close_fds=True, pass_fds=pass_fds,
        )
        containment.bind_root(process.pid, process.pid)
        audit_child_fd_inheritance(
            process.pid, pass_fds, expected_binding=process_binding(process.pid),
            forbidden_descriptors=forbidden_descriptors, parent_identities=parent_identities,
        )
        try:
            stdout, stderr = bounded_exchange(process, request, timeout_seconds, containment=containment)
            result = RawExecution(process.returncode, False, bool(process.returncode is not None and process.returncode < 0), False, stdout, stderr, None)
        except subprocess.TimeoutExpired:
            result = RawExecution(process.poll(), True, False, False, b"", b"", "runtime timed out")
        except (OSError, RunnerError) as exc:
            result = RawExecution(process.poll(), False, bool(process.poll() is not None and process.poll() < 0), False, b"", b"", str(exc))
    except (OSError, RunnerError, ValueError) as exc:
        result = RawExecution(process.poll() if process is not None else None, False, False, False, b"", b"", str(exc))
    finally:
        proven = False
        if process is not None and containment is not None:
            proven = containment.terminate_and_reap(process)
            close_process_streams(process)
        elif process is None and containment is not None:
            proven = containment.restore_after_launch_failure()
        if result is None:
            result = RawExecution(None, False, False, False, b"", b"", "runtime launch produced no result")
        if not proven:
            result = RawExecution(result.exit_code, result.timed_out, result.crashed, False, result.stdout, result.stderr, result.error or "runtime Linux containment/reaping was not proven")
        else:
            result = RawExecution(result.exit_code, result.timed_out, result.crashed, True, result.stdout, result.stderr, result.error)
    return result


def _strict_json(data: bytes, label: str) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=strict_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunnerError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise RunnerError(f"{label} is not an object")
    return document


def _ipc_send(sock: socket.socket, document: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(dict(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunnerError("worker IPC document is not canonical JSON") from exc
    if not payload or len(payload) > MAX_OUTPUT:
        raise RunnerError("worker IPC document exceeds its fixed bound")
    try:
        sent = sock.send(payload)
    except OSError as exc:
        raise RunnerError("cannot send worker IPC frame") from exc
    if sent != len(payload):
        raise RunnerError("worker IPC short write")


def _close_rights(data: bytes) -> None:
    values = array.array("i")
    values.frombytes(data[:len(data) - len(data) % values.itemsize])
    for descriptor in set(values):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _ipc_recv(sock: socket.socket) -> tuple[dict[str, Any], tuple[int, int, int]]:
    ancillary = socket.CMSG_SPACE(struct.calcsize("3i")) + socket.CMSG_SPACE(16 * struct.calcsize("i"))
    try:
        payload, control, flags, _address = sock.recvmsg(MAX_OUTPUT + 1, ancillary)
    except OSError as exc:
        raise RunnerError("cannot receive worker IPC frame") from exc
    credentials: tuple[int, int, int] | None = None
    invalid = False
    for level, kind, data in control:
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and credentials is None and len(data) >= struct.calcsize("3i"):
            credentials = struct.unpack("3i", data[:struct.calcsize("3i")])
        elif level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            _close_rights(data)
            invalid = True
        else:
            invalid = True
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not payload or len(payload) > MAX_OUTPUT:
        raise RunnerError("worker IPC frame is truncated or oversized")
    if invalid or credentials is None:
        raise RunnerError("worker IPC ancillary data is malformed")
    return _strict_json(payload, "worker IPC frame"), credentials


def _decode_b64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise RunnerError(f"{label} is not base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise RunnerError(f"{label} is invalid base64") from exc


def _failure(sock: socket.socket, challenge: str | None, order: int | None, error: str) -> None:
    document: dict[str, Any] = {"kind": "failure", "error": error[:1024]}
    if challenge is not None:
        document["challenge"] = challenge
    if order is not None:
        document["order"] = order
    _ipc_send(sock, document)


def worker_main(args: argparse.Namespace) -> int:
    sock = socket.socket(fileno=args.controller_fd)
    challenge: str | None = None
    try:
        if not sys.flags.isolated:
            raise RunnerError("sealed G1 worker requires Python isolated mode")
        own_binding = process_binding(os.getpid())
        _ipc_send(sock, {"kind": "ready", "binding": own_binding})
        start, credentials = _ipc_recv(sock)
        if credentials != (args.controller_pid, os.getuid(), os.getgid()) or start.get("kind") != "start":
            raise RunnerError("worker start frame is not from its controller")
        if start.get("row_id") != args.row or start.get("target") != args.target or not isinstance(start.get("challenge"), str) or len(start["challenge"]) != 64:
            raise RunnerError("worker start frame row/target/challenge is invalid")
        challenge = start["challenge"]
        loader_fd, executable_fd = int(args.loader_fd), int(args.executable_fd)
        library_fds = tuple(int(value) for value in args.library_fd)
        for expected_order in range(EXPECTED_CASE_COUNT):
            command, credentials = _ipc_recv(sock)
            if credentials != (args.controller_pid, os.getuid(), os.getgid()) or command.get("kind") != "case" or command.get("challenge") != challenge or command.get("order") != expected_order:
                raise RunnerError("worker case frame is not the expected one-shot controller command")
            request = _decode_b64(command.get("request_b64"), "controller request")
            execution = execute_raw_runtime(
                loader_fd=loader_fd, executable_fd=executable_fd, runtime_fds=library_fds,
                target=args.target, physical_hip_index=args.physical_hip_index, logical_device_index=0,
                request=request, timeout_seconds=float(args.timeout_seconds), forbidden_descriptors=(args.controller_fd,),
            )
            if execution.exit_code != 0 or execution.timed_out or execution.crashed or not execution.cleanup_proven or execution.error is not None:
                error = execution.error or "runtime process failed"
                print(f"semantic G1 worker runtime failure: {error}", file=sys.stderr, flush=True)
                _failure(sock, challenge, expected_order, error)
                return 2
            _ipc_send(sock, {
                "kind": "raw-case", "challenge": challenge, "order": expected_order,
                "response_b64": base64.b64encode(execution.stdout).decode("ascii"),
                "stderr_b64": base64.b64encode(execution.stderr).decode("ascii"),
            })
        finish, credentials = _ipc_recv(sock)
        if credentials != (args.controller_pid, os.getuid(), os.getgid()) or finish != {"kind": "finish", "challenge": challenge}:
            raise RunnerError("worker finish frame is malformed")
        _ipc_send(sock, {"kind": "done", "challenge": challenge, "case_count": EXPECTED_CASE_COUNT})
        return 0
    except (RunnerError, OSError, ValueError, TypeError) as exc:
        try:
            _failure(sock, challenge, None, str(exc))
        except (RunnerError, OSError):
            pass
        return 2
    finally:
        try:
            sock.close()
        except OSError:
            pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", action="store_true")
    result.add_argument("--controller-fd", type=int)
    result.add_argument("--controller-pid", type=int)
    result.add_argument("--row", choices=ROWS)
    result.add_argument("--target", choices=TARGETS)
    result.add_argument("--physical-hip-index", type=int)
    result.add_argument("--timeout-seconds", type=float)
    result.add_argument("--loader-fd", type=int)
    result.add_argument("--executable-fd", type=int)
    result.add_argument("--library-fd", action="append", type=int, default=[])
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.worker:
        print("semantic RMSNorm G1 runner: FAIL-CLOSED: controller worker mode is required", file=sys.stderr)
        return 2
    required = (args.controller_fd, args.controller_pid, args.row, args.target, args.physical_hip_index, args.timeout_seconds, args.loader_fd, args.executable_fd)
    if any(value is None for value in required) or len(args.library_fd) < 1 or args.row != f"rmsnorm-semantic-g1-{args.target}":
        print("semantic RMSNorm G1 runner: FAIL-CLOSED: incomplete controller worker arguments", file=sys.stderr)
        return 2
    try:
        return worker_main(args)
    except (RunnerError, OSError, ValueError, TypeError) as exc:
        print(f"semantic RMSNorm G1 runner: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
