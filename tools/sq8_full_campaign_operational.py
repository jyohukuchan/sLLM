#!/usr/bin/env python3
"""Read-only operational preflight for the production SQ8 campaign.

The orchestration layer supplies every external dependency.  This module only
performs bounded GETs, bounded read-only inspection commands, pinned directory
metadata reads, and the existing worker-acceptance resource probes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast


MAX_COMMAND_STDOUT_BYTES = 8 << 20
MAX_COMMAND_STDERR_BYTES = 64 << 10
MAX_HTTP_BODY_BYTES = 4096
COMMAND_TIMEOUT_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 10.0
MINIMUM_ACTIVE_AGE_NS = 900_000_000_000
SYSTEMD_RESTART_USEC = 10_000_000
SYSTEMD_START_LIMIT_INTERVAL_USEC = 900_000_000
SYSTEMD_START_LIMIT_BURST = 3
GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545
AMD_SMI_BIN = "/opt/rocm/bin/amd-smi"
SYSTEMCTL_BIN = "/usr/bin/systemctl"
DOCKER_BIN = "/usr/bin/docker"
SUDO_BIN = "/usr/bin/sudo"
NSENTER_BIN = "/usr/bin/nsenter"
PYTHON_BIN = "/usr/bin/python3"
EXECUTION_UID = 1000
EXECUTION_GID = 1000
OPENWEBUI_CONTAINER_NAME = "open-webui"
OPENWEBUI_IMAGE_ID = (
    "sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
OPENWEBUI_NETWORK_NAME = "open-webui-network"
OPENWEBUI_NETWORK_ID = (
    "79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8"
)
COMMAND_ENVIRONMENT = (
    ("HOME", "/"),
    ("LANG", "C"),
    ("LC_ALL", "C"),
    ("PATH", "/usr/bin:/bin:/opt/rocm/bin"),
    ("SYSTEMD_COLORS", "0"),
    ("SYSTEMD_PAGER", ""),
)

GATEWAY_READY_BODY = b'{"status":"ready"}'
OPENWEBUI_HEALTH_BODY = b'{"status":true}'
GATEWAY_NAMESPACE_OUTPUT = b'200\n{"status":"ready"}'
GATEWAY_NAMESPACE_SOURCE = (
    "import http.client,sys;"
    "c=http.client.HTTPConnection('172.20.0.1',8000,timeout=5.0);"
    "c.request('GET','/readyz',headers={'Accept':'application/json',"
    "'Connection':'close'});r=c.getresponse();b=r.read(4097);c.close();"
    "sys.stdout.buffer.write(str(r.status).encode('ascii')+b'\\n'+b)"
)

CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
CONTAINER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SERVICE_UNIT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,254}\.service\Z")


class OperationalError(RuntimeError):
    """One read-only production prerequisite differs from the contract."""


def fail(message: str) -> NoReturn:
    raise OperationalError(message)


@dataclasses.dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    body: bytes


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
        )


@dataclasses.dataclass(frozen=True)
class ObserverPathSnapshot:
    pinned_parent: FileIdentity
    named_parent: FileIdentity | None
    child: FileIdentity | None


@dataclasses.dataclass(frozen=True)
class SystemdSnapshot:
    main_pid: int
    n_restarts: int
    active_enter_monotonic_us: int
    captured_monotonic_ns: int
    active_age_ns: int
    restart: str
    restart_usec: int
    start_limit_interval_usec: int
    start_limit_burst: int

    def stable_identity(self) -> tuple[int, int, int, str, int, int, int]:
        return (
            self.main_pid,
            self.n_restarts,
            self.active_enter_monotonic_us,
            self.restart,
            self.restart_usec,
            self.start_limit_interval_usec,
            self.start_limit_burst,
        )


@dataclasses.dataclass(frozen=True)
class ContainerSnapshot:
    container_id: str
    image_id: str
    pid: int
    restart_count: int
    started_at: str
    network_ids: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class GpuIsolationSnapshot:
    worker_pid: int
    worker_starttime_ticks: int
    amd_vram_bytes: int
    kfd_vram_bytes: int
    positive_kfd_pids: tuple[int, ...]
    amd_list_sha256: str
    kfd_attempt_count: int

    def stable_identity(self) -> tuple[int, int, int, int, tuple[int, ...], str]:
        return (
            self.worker_pid,
            self.worker_starttime_ticks,
            self.amd_vram_bytes,
            self.kfd_vram_bytes,
            self.positive_kfd_pids,
            self.amd_list_sha256,
        )


@dataclasses.dataclass(frozen=True)
class OperationalExpectation:
    service_unit: str
    gateway_pid: int
    worker_pid: int
    container_name: str
    container_id: str
    image_id: str
    network_name: str
    network_id: str
    gateway_ready_url: str
    openwebui_health_url: str
    observer_socket: Path
    observer_parent_uid: int
    observer_parent_gid: int
    observer_parent_mode: int = 0o750


@dataclasses.dataclass(frozen=True)
class OperationalSnapshot:
    systemd: SystemdSnapshot
    container: ContainerSnapshot
    gpu: GpuIsolationSnapshot
    observer_parent: FileIdentity
    gateway_ready: HttpResponse
    openwebui_health: HttpResponse


class CommandReader(Protocol):
    def run(
        self,
        arguments: tuple[str, ...],
        *,
        label: str,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
    ) -> bytes: ...


class HttpReader(Protocol):
    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> HttpResponse: ...


class GatewayNamespaceReader(Protocol):
    def get(
        self,
        container_pid: int,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> HttpResponse: ...


class ObserverPathHandle(Protocol):
    def snapshot(self) -> ObserverPathSnapshot: ...

    def close(self) -> None: ...


class ObserverPathReader(Protocol):
    def open_path(self, path: Path) -> ObserverPathHandle: ...


class GpuIsolationReader(Protocol):
    def capture(self, worker_pid: int) -> GpuIsolationSnapshot: ...


@dataclasses.dataclass(frozen=True)
class OperationalDependencies:
    commands: CommandReader
    http: HttpReader
    gateway_http: GatewayNamespaceReader
    observer_paths: ObserverPathReader
    gpu: GpuIsolationReader
    monotonic_ns: Callable[[], int]


@dataclasses.dataclass(frozen=True)
class BoundedReadOnlyCommandReader:
    """Execute only caller-pinned, read-only command vectors."""

    allowed_commands: frozenset[tuple[str, ...]]
    maximum_timeout_seconds: float = 30.0
    maximum_output_bytes: int = MAX_COMMAND_STDOUT_BYTES

    def run(
        self,
        arguments: tuple[str, ...],
        *,
        label: str,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
    ) -> bytes:
        if (
            type(arguments) is not tuple
            or not arguments
            or any(type(item) is not str or not item for item in arguments)
            or arguments not in self.allowed_commands
            or not _is_read_only_command(arguments)
        ):
            fail("command is not in the pinned read-only allowlist")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= self.maximum_timeout_seconds
            or type(maximum_stdout_bytes) is not int
            or not 0 < maximum_stdout_bytes <= self.maximum_output_bytes
        ):
            fail("bounded command limits differ")
        return _run_bounded_read_only_process(
            arguments,
            label=label,
            timeout_seconds=float(timeout_seconds),
            maximum_stdout_bytes=maximum_stdout_bytes,
        )


def _terminate_probe_process(process: subprocess.Popen[bytes]) -> str | None:
    cleanup_errors: list[str] = []
    try:
        running = process.poll() is None
    except (OSError, subprocess.SubprocessError):
        running = True
        cleanup_errors.append("poll failed")
    if running:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_errors.append("process-group kill failed")
    try:
        process.wait(timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        cleanup_errors.append("wait failed")
    return "; ".join(cleanup_errors) or None


def _close_probe_resources(
    process: subprocess.Popen[bytes], selector: selectors.BaseSelector
) -> str | None:
    cleanup_errors: list[str] = []
    termination_error = _terminate_probe_process(process)
    if termination_error is not None:
        cleanup_errors.append(termination_error)
    try:
        selector.close()
    except (OSError, ValueError):
        cleanup_errors.append("selector close failed")
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            cleanup_errors.append("pipe close failed")
    return "; ".join(cleanup_errors) or None


def _run_bounded_read_only_process(
    arguments: tuple[str, ...],
    *,
    label: str,
    timeout_seconds: float,
    maximum_stdout_bytes: int,
    preserve_controlling_tty: bool = False,
) -> bytes:
    if type(preserve_controlling_tty) is not bool:
        fail(f"{label} controlling TTY binding differs")
    try:
        process = subprocess.Popen(
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=not preserve_controlling_tty,
            process_group=0 if preserve_controlling_tty else None,
            env=dict(COMMAND_ENVIRONMENT),
        )
    except (OSError, subprocess.SubprocessError):
        fail(f"failed to execute {label}")
    if process.stdout is None or process.stderr is None:
        cleanup_error = _terminate_probe_process(process)
        if cleanup_error is not None:
            fail(f"{label} pipe setup and cleanup failed")
        fail(f"{label} lacks bounded output pipes")

    selector = selectors.DefaultSelector()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_bytes = 0
    stderr_bytes = 0
    try:
        for stream, channel in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, channel)
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fail(f"{label} timed out")
            events = selector.select(remaining)
            if not events:
                if time.monotonic() >= deadline:
                    fail(f"{label} timed out")
                continue
            for key, _mask in events:
                channel = cast(str, key.data)
                maximum = (
                    maximum_stdout_bytes
                    if channel == "stdout"
                    else MAX_COMMAND_STDERR_BYTES
                )
                observed = stdout_bytes if channel == "stdout" else stderr_bytes
                try:
                    chunk = os.read(key.fd, min(64 << 10, maximum - observed + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if channel == "stdout":
                    stdout_chunks.append(chunk)
                    stdout_bytes += len(chunk)
                    if stdout_bytes > maximum_stdout_bytes:
                        fail(f"{label} stdout exceeds its byte bound")
                else:
                    stderr_chunks.append(chunk)
                    stderr_bytes += len(chunk)
                    if stderr_bytes > MAX_COMMAND_STDERR_BYTES:
                        fail(f"{label} stderr exceeds its byte bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            fail(f"{label} timed out")
        try:
            return_code = process.wait(timeout=max(0.0, remaining))
        except subprocess.TimeoutExpired:
            fail(f"{label} timed out")
        if return_code != 0:
            fail(f"{label} exited {return_code}")
        if stderr_chunks:
            fail(f"{label} emitted stderr")
        result = b"".join(stdout_chunks)
    except BaseException as primary:
        cleanup_error = _close_probe_resources(process, selector)
        if cleanup_error is not None:
            primary.add_note(f"probe cleanup also failed: {cleanup_error}")
        raise
    cleanup_error = _close_probe_resources(process, selector)
    if cleanup_error is not None:
        fail(f"{label} cleanup failed")
    return result


@dataclasses.dataclass(frozen=True)
class BoundedHttpReader:
    """Issue direct HTTP GETs only to caller-pinned URLs."""

    allowed_urls: frozenset[str]
    maximum_timeout_seconds: float = 30.0
    maximum_response_bytes: int = MAX_HTTP_BODY_BYTES

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> HttpResponse:
        if url not in self.allowed_urls:
            fail("HTTP URL is not in the pinned read-only allowlist")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= self.maximum_timeout_seconds
            or type(maximum_body_bytes) is not int
            or not 0 < maximum_body_bytes <= self.maximum_response_bytes
        ):
            fail("bounded HTTP limits differ")
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/")
        ):
            fail("HTTP probe URL is not one direct absolute HTTP URL")
        connection = http.client.HTTPConnection(
            parsed.hostname,
            parsed.port,
            timeout=float(timeout_seconds),
        )
        try:
            connection.request(
                "GET",
                parsed.path,
                headers={"Accept": "application/json", "Connection": "close"},
            )
            response = connection.getresponse()
            body = response.read(maximum_body_bytes + 1)
            status = response.status
        except (OSError, http.client.HTTPException):
            fail("bounded HTTP GET failed")
        finally:
            connection.close()
        if len(body) > maximum_body_bytes:
            fail("HTTP response body exceeds its byte bound")
        return HttpResponse(url, status, body)


@dataclasses.dataclass(frozen=True)
class ProductionGatewayNamespaceReader:
    """Read gateway readiness through an existing container network namespace."""

    def get(
        self,
        container_pid: int,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> HttpResponse:
        if type(container_pid) is not int or container_pid <= 0:
            fail("gateway namespace PID differs")
        if timeout_seconds != HTTP_TIMEOUT_SECONDS:
            fail("gateway namespace timeout differs")
        if maximum_body_bytes != MAX_HTTP_BODY_BYTES:
            fail("gateway namespace body bound differs")
        arguments = (
            SUDO_BIN,
            "-n",
            NSENTER_BIN,
            "--target",
            str(container_pid),
            "--net",
            "--setgid",
            str(EXECUTION_GID),
            "--setuid",
            str(EXECUTION_UID),
            PYTHON_BIN,
            "-I",
            "-c",
            GATEWAY_NAMESPACE_SOURCE,
        )
        raw = _run_bounded_read_only_process(
            arguments,
            label="gateway namespace readiness GET",
            timeout_seconds=HTTP_TIMEOUT_SECONDS,
            maximum_stdout_bytes=len(GATEWAY_NAMESPACE_OUTPUT),
            preserve_controlling_tty=True,
        )
        if raw != GATEWAY_NAMESPACE_OUTPUT:
            fail("gateway namespace readiness output differs")
        return HttpResponse("http://172.20.0.1:8000/readyz", 200, GATEWAY_READY_BODY)


class _OsObserverPathHandle:
    def __init__(self, path: Path, parent_fd: int) -> None:
        self._path = path
        self._parent_fd = parent_fd

    def snapshot(self) -> ObserverPathSnapshot:
        if self._parent_fd < 0:
            fail("observer path handle is closed")
        try:
            pinned = FileIdentity.from_stat(os.fstat(self._parent_fd))
            try:
                named = FileIdentity.from_stat(
                    os.stat(self._path.parent, follow_symlinks=False)
                )
            except FileNotFoundError:
                named = None
            try:
                child = FileIdentity.from_stat(
                    os.stat(
                        self._path.name,
                        dir_fd=self._parent_fd,
                        follow_symlinks=False,
                    )
                )
            except FileNotFoundError:
                child = None
        except OperationalError:
            raise
        except OSError:
            fail("failed to inspect the pinned observer path")
        return ObserverPathSnapshot(pinned, named, child)

    def close(self) -> None:
        if self._parent_fd < 0:
            return
        descriptor = self._parent_fd
        self._parent_fd = -1
        try:
            os.close(descriptor)
        except OSError:
            fail("failed to close the observer parent handle")


@dataclasses.dataclass(frozen=True)
class OsObserverPathReader:
    """Pin an existing observer parent and inspect a missing child by dirfd."""

    def open_path(self, path: Path) -> ObserverPathHandle:
        absolute = _observer_path(path)
        if not hasattr(os, "O_NOFOLLOW"):
            fail("O_NOFOLLOW is required for observer path inspection")
        try:
            descriptor = os.open(
                absolute.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        except OSError:
            fail("failed to pin the observer parent directory")
        return _OsObserverPathHandle(absolute, descriptor)


LegacyCommandRunner = Callable[[list[str], str], bytes]
ProbeContextFactory = Callable[[str, Path, Path, LegacyCommandRunner], object]
ResourceCapture = Callable[[object, int], tuple[dict[str, Any], int, int]]


@dataclasses.dataclass(frozen=True)
class WorkerAcceptanceGpuReader:
    """Adapt the established worker-acceptance AMD, /proc, and KFD probes."""

    commands: CommandReader
    amd_smi: str
    proc_root: Path
    kfd_proc_root: Path
    parse_amd_smi_list: Callable[[bytes], None]
    probe_context_factory: ProbeContextFactory
    capture_resource_sample: ResourceCapture

    @classmethod
    def from_collector_module(
        cls,
        module: object,
        commands: CommandReader,
        *,
        amd_smi: str = AMD_SMI_BIN,
        proc_root: Path = Path("/proc"),
        kfd_proc_root: Path = Path("/sys/class/kfd/kfd/proc"),
    ) -> WorkerAcceptanceGpuReader:
        if amd_smi != AMD_SMI_BIN:
            fail("production AMD SMI executable differs")
        for name, expected in (
            ("GPU_INDEX", GPU_INDEX),
            ("GPU_BDF", GPU_BDF),
            ("GPU_UUID", GPU_UUID),
            ("KFD_GPU_ID", KFD_GPU_ID),
        ):
            if getattr(module, name, None) != expected:
                fail("worker acceptance GPU identity constants differ")
        parse_list = getattr(module, "parse_amd_smi_list", None)
        context_factory = getattr(module, "ProbeContext", None)
        capture_sample = getattr(module, "capture_resource_sample", None)
        if not all(
            callable(item) for item in (parse_list, context_factory, capture_sample)
        ):
            fail("worker acceptance resource probe API is incomplete")
        return cls(
            commands,
            amd_smi,
            proc_root,
            kfd_proc_root,
            cast(Callable[[bytes], None], parse_list),
            cast(ProbeContextFactory, context_factory),
            cast(ResourceCapture, capture_sample),
        )

    def capture(self, worker_pid: int) -> GpuIsolationSnapshot:
        if type(worker_pid) is not int or worker_pid <= 0:
            fail("GPU isolation worker PID is invalid")

        def legacy_runner(arguments: list[str], label: str) -> bytes:
            return self.commands.run(
                tuple(arguments),
                label=label,
                timeout_seconds=COMMAND_TIMEOUT_SECONDS,
                maximum_stdout_bytes=MAX_COMMAND_STDOUT_BYTES,
            )

        try:
            list_raw = legacy_runner([self.amd_smi, "list", "--json"], "amd-smi list")
            self.parse_amd_smi_list(list_raw)
            context = self.probe_context_factory(
                self.amd_smi,
                self.proc_root,
                self.kfd_proc_root,
                legacy_runner,
            )
            captured, rss_bytes, amd_vram = self.capture_resource_sample(
                context, worker_pid
            )
        except OperationalError:
            raise
        except Exception as error:
            raise OperationalError("R9700 isolation probe failed") from error
        return _gpu_snapshot(
            captured,
            rss_bytes,
            amd_vram,
            worker_pid,
            hashlib.sha256(list_raw).hexdigest(),
        )


def load_worker_acceptance_gpu_reader(
    commands: CommandReader,
    *,
    amd_smi: str = AMD_SMI_BIN,
    proc_root: Path = Path("/proc"),
    kfd_proc_root: Path = Path("/sys/class/kfd/kfd/proc"),
) -> WorkerAcceptanceGpuReader:
    """Load the existing hyphenated collector without running a probe."""

    path = Path(__file__).resolve().with_name("run-sq8-worker-acceptance.py")
    name = "_sq8_operational_worker_acceptance"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail("failed to load the worker acceptance collector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise OperationalError(
            "failed to load the worker acceptance collector"
        ) from error
    return WorkerAcceptanceGpuReader.from_collector_module(
        module,
        commands,
        amd_smi=amd_smi,
        proc_root=proc_root,
        kfd_proc_root=kfd_proc_root,
    )


def _strict_json(raw: bytes, label: str) -> Any:
    if type(raw) is not bytes or not raw or len(raw) > MAX_COMMAND_STDOUT_BYTES:
        fail(f"{label} size differs")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            fail(f"{label} contains a non-finite number")
        return parsed

    def reject_constant(_value: str) -> NoReturn:
        fail(f"{label} contains a non-finite constant")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except OperationalError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        fail(f"{label} is not strict UTF-8 JSON")


def _key_value_lines(raw: bytes, label: str) -> dict[str, str]:
    if type(raw) is not bytes or not raw or len(raw) > 4096:
        fail(f"{label} size differs")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        fail(f"{label} is not UTF-8")
    values: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            fail(f"{label} contains malformed or duplicate fields")
        values[key] = value
    return values


def _decimal(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        fail(f"{label} is not canonical decimal")
    parsed = int(value, 10)
    if value != str(parsed) or parsed < minimum:
        fail(f"{label} is out of range")
    return parsed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer in range")
    return value


def _text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        fail(f"{label} is not one bounded non-empty string")
    return value


def _systemd_duration_us(value: str, label: str) -> int:
    match = re.fullmatch(r"([0-9]+)(us|ms|s|min|h)", value)
    if match is None or match.group(1) != str(int(match.group(1), 10)):
        fail(f"{label} is not one exact systemd duration")
    multiplier = {
        "us": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60_000_000,
        "h": 3_600_000_000,
    }[match.group(2)]
    return int(match.group(1), 10) * multiplier


def _systemd_command(unit: str) -> tuple[str, ...]:
    return (
        SYSTEMCTL_BIN,
        "show",
        unit,
        "--property=ActiveState",
        "--property=SubState",
        "--property=Result",
        "--property=MainPID",
        "--property=NRestarts",
        "--property=Restart",
        "--property=RestartUSec",
        "--property=StartLimitIntervalUSec",
        "--property=StartLimitBurst",
        "--property=ActiveEnterTimestampMonotonic",
        "--no-pager",
    )


def _capture_systemd(
    commands: CommandReader,
    unit: str,
    expected_pid: int,
    monotonic_ns: Callable[[], int],
) -> SystemdSnapshot:
    raw = commands.run(
        _systemd_command(unit),
        label="SQ8 gateway systemd preflight",
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        maximum_stdout_bytes=4096,
    )
    values = _key_value_lines(raw, "systemd preflight")
    expected_fields = {
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "NRestarts",
        "Restart",
        "RestartUSec",
        "StartLimitIntervalUSec",
        "StartLimitBurst",
        "ActiveEnterTimestampMonotonic",
    }
    if set(values) != expected_fields:
        fail("systemd preflight field set differs")
    if (
        values["ActiveState"] != "active"
        or values["SubState"] != "running"
        or values["Result"] != "success"
    ):
        fail("systemd service is not active, running, and successful")
    main_pid = _decimal(values["MainPID"], "systemd MainPID", minimum=1)
    if main_pid != expected_pid:
        fail("systemd MainPID differs from the captured gateway")
    n_restarts = _decimal(values["NRestarts"], "systemd NRestarts")
    active_enter_us = _decimal(
        values["ActiveEnterTimestampMonotonic"],
        "systemd active-enter monotonic timestamp",
        minimum=1,
    )
    restart_usec = _systemd_duration_us(values["RestartUSec"], "RestartUSec")
    start_limit_usec = _systemd_duration_us(
        values["StartLimitIntervalUSec"], "StartLimitIntervalUSec"
    )
    start_limit_burst = _decimal(
        values["StartLimitBurst"], "StartLimitBurst", minimum=1
    )
    if (
        values["Restart"] != "on-failure"
        or restart_usec != SYSTEMD_RESTART_USEC
        or start_limit_usec != SYSTEMD_START_LIMIT_INTERVAL_USEC
        or start_limit_burst != SYSTEMD_START_LIMIT_BURST
    ):
        fail("systemd restart or start-limit policy differs")
    captured_ns = monotonic_ns()
    if type(captured_ns) is not int or captured_ns < 0:
        fail("monotonic clock returned an invalid value")
    active_enter_ns = active_enter_us * 1000
    if captured_ns < active_enter_ns:
        fail("systemd active-enter timestamp is in the future")
    age_ns = captured_ns - active_enter_ns
    if age_ns < MINIMUM_ACTIVE_AGE_NS:
        fail("systemd service has not remained active for 900 seconds")
    return SystemdSnapshot(
        main_pid,
        n_restarts,
        active_enter_us,
        captured_ns,
        age_ns,
        values["Restart"],
        restart_usec,
        start_limit_usec,
        start_limit_burst,
    )


def _docker_command(name: str) -> tuple[str, ...]:
    return (DOCKER_BIN, "container", "inspect", name)


def _is_read_only_command(arguments: tuple[str, ...]) -> bool:
    if len(arguments) >= 3 and arguments[:2] == (SYSTEMCTL_BIN, "show"):
        unit = arguments[2]
        return SERVICE_UNIT_RE.fullmatch(
            unit
        ) is not None and arguments == _systemd_command(unit)
    if len(arguments) == 4 and arguments[:3] == (
        DOCKER_BIN,
        "container",
        "inspect",
    ):
        return CONTAINER_NAME_RE.fullmatch(arguments[3]) is not None
    if not arguments or arguments[0] != AMD_SMI_BIN:
        return False
    return arguments[1:] in {
        ("list", "--json"),
        ("process", "--gpu", str(GPU_INDEX), "--general", "--json"),
    }


def production_read_only_commands(
    expectation: OperationalExpectation,
    *,
    amd_smi: str = AMD_SMI_BIN,
) -> frozenset[tuple[str, ...]]:
    """Return the complete immutable command set used by this preflight."""

    _validate_expectation(expectation)
    if amd_smi != AMD_SMI_BIN:
        fail("production AMD SMI executable differs")
    commands = frozenset(
        {
            _systemd_command(expectation.service_unit),
            _docker_command(expectation.container_name),
            (amd_smi, "list", "--json"),
            (
                amd_smi,
                "process",
                "--gpu",
                str(GPU_INDEX),
                "--general",
                "--json",
            ),
        }
    )
    if any(not _is_read_only_command(item) for item in commands):
        fail("production command set contains a non-read-only command")
    return commands


def production_container_discovery_commands() -> frozenset[tuple[str, ...]]:
    """Return the sole command allowed before the container ID is known."""

    command = _docker_command(OPENWEBUI_CONTAINER_NAME)
    if not _is_read_only_command(command):
        fail("production container discovery command is not read-only")
    return frozenset({command})


def _capture_container_identity(
    commands: CommandReader,
    *,
    container_name: str,
    container_id: str | None,
    image_id: str,
    network_name: str,
    network_id: str,
) -> ContainerSnapshot:
    raw = commands.run(
        _docker_command(container_name),
        label="OpenWebUI container inspection",
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        maximum_stdout_bytes=MAX_COMMAND_STDOUT_BYTES,
    )
    value = _strict_json(raw, "OpenWebUI container inspection")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        fail("OpenWebUI container inspection shape differs")
    container = cast(dict[str, Any], value[0])
    observed_container_id = container.get("Id")
    if (
        type(observed_container_id) is not str
        or CONTAINER_ID_RE.fullmatch(observed_container_id) is None
        or (container_id is not None and observed_container_id != container_id)
        or container.get("Name") != f"/{container_name}"
        or container.get("Image") != image_id
    ):
        fail("OpenWebUI container or image content identity differs")
    state = container.get("State")
    if type(state) is not dict:
        fail("OpenWebUI container state is absent")
    health = state.get("Health")
    if (
        state.get("Status") != "running"
        or state.get("Running") is not True
        or state.get("Restarting") is not False
        or state.get("OOMKilled") is not False
        or state.get("Dead") is not False
        or type(health) is not dict
        or health.get("Status") != "healthy"
    ):
        fail("OpenWebUI container is not running and healthy")
    pid = _integer(state.get("Pid"), "OpenWebUI container PID", minimum=1)
    started_at = _text(state.get("StartedAt"), "OpenWebUI StartedAt", maximum=128)
    restart_count = _integer(container.get("RestartCount"), "OpenWebUI RestartCount")
    network_settings = container.get("NetworkSettings")
    networks = (
        network_settings.get("Networks") if type(network_settings) is dict else None
    )
    if type(networks) is not dict or set(networks) != {network_name}:
        fail("OpenWebUI Docker network attachment set differs")
    attachment = networks[network_name]
    if type(attachment) is not dict or attachment.get("NetworkID") != network_id:
        fail("OpenWebUI Docker network content identity differs")
    return ContainerSnapshot(
        observed_container_id,
        image_id,
        pid,
        restart_count,
        started_at,
        ((network_name, network_id),),
    )


def _capture_container(
    commands: CommandReader, expectation: OperationalExpectation
) -> ContainerSnapshot:
    return _capture_container_identity(
        commands,
        container_name=expectation.container_name,
        container_id=expectation.container_id,
        image_id=expectation.image_id,
        network_name=expectation.network_name,
        network_id=expectation.network_id,
    )


def discover_production_openwebui_container(
    commands: CommandReader,
) -> ContainerSnapshot:
    """Discover the current fixed production OpenWebUI container identity."""

    return _capture_container_identity(
        commands,
        container_name=OPENWEBUI_CONTAINER_NAME,
        container_id=None,
        image_id=OPENWEBUI_IMAGE_ID,
        network_name=OPENWEBUI_NETWORK_NAME,
        network_id=OPENWEBUI_NETWORK_ID,
    )


def _observer_path(path: Path) -> Path:
    if not isinstance(path, os.PathLike):
        fail("observer socket path is invalid")
    raw = os.fspath(path)
    candidate = Path(raw)
    if (
        not candidate.is_absolute()
        or raw != os.fspath(candidate)
        or candidate.name in {"", ".", ".."}
        or ".." in candidate.parts
        or candidate.parent == candidate
    ):
        fail("observer socket path is not one canonical absolute child path")
    return candidate


def _require_observer_snapshot(
    snapshot: ObserverPathSnapshot,
    expectation: OperationalExpectation,
) -> FileIdentity:
    parent = snapshot.pinned_parent
    if snapshot.named_parent != parent:
        fail("observer parent pathname differs from its pinned directory")
    if (
        snapshot.child is not None
        or not stat.S_ISDIR(parent.mode)
        or stat.S_ISLNK(parent.mode)
        or stat.S_IMODE(parent.mode) != expectation.observer_parent_mode
        or parent.uid != expectation.observer_parent_uid
        or parent.gid != expectation.observer_parent_gid
        or parent.links < 2
    ):
        fail("observer parent identity or required missing path differs")
    return parent


def _gpu_snapshot(
    captured: dict[str, Any],
    rss_bytes: int,
    amd_vram: int,
    worker_pid: int,
    amd_list_sha256: str,
) -> GpuIsolationSnapshot:
    if type(captured) is not dict:
        fail("worker resource capture is not an object")
    worker = captured.get("worker")
    gpu = captured.get("gpu")
    if type(worker) is not dict or type(gpu) is not dict:
        fail("worker resource capture lacks worker or GPU data")
    start_before = _integer(
        worker.get("starttime_ticks_before"), "worker starttime", minimum=1
    )
    start_after = _integer(
        worker.get("starttime_ticks_after"), "worker final starttime", minimum=1
    )
    if (
        worker.get("pid") != worker_pid
        or start_before != start_after
        or Path(_text(worker.get("exe_target"), "worker executable")).name
        != "ullm-sq8-worker"
        or worker.get("children") != []
        or worker.get("vmrss_bytes") != rss_bytes
        or type(rss_bytes) is not int
        or rss_bytes <= 0
    ):
        fail("worker /proc identity changed or differs")
    if (
        gpu.get("index") != GPU_INDEX
        or gpu.get("bdf") != GPU_BDF
        or gpu.get("uuid") != GPU_UUID
        or gpu.get("kfd_gpu_id") != KFD_GPU_ID
        or gpu.get("worker_pid") != worker_pid
        or gpu.get("mem_usage_unit") != "B"
        or gpu.get("mem_usage_value") != amd_vram
        or type(amd_vram) is not int
        or amd_vram <= 0
    ):
        fail("AMD SMI R9700 worker capture differs")
    kfd = gpu.get("kfd_snapshot")
    if type(kfd) is not dict:
        fail("KFD snapshot is absent")
    attempts = _integer(kfd.get("attempt_count"), "KFD attempt count", minimum=1)
    processes = kfd.get("processes")
    before = kfd.get("before_identities")
    after = kfd.get("after_identities")
    attempt_values = kfd.get("attempts")
    if (
        type(processes) is not list
        or type(before) is not list
        or type(after) is not list
        or type(attempt_values) is not list
        or len(attempt_values) != attempts
        or not attempt_values
        or type(attempt_values[-1]) is not dict
        or attempt_values[-1].get("outcome") != "stable"
    ):
        fail("KFD snapshot structure differs")
    acquisition_started = _integer(
        kfd.get("acquisition_started_monotonic_ns"), "KFD acquisition start"
    )
    acquisition_completed = _integer(
        kfd.get("acquisition_completed_monotonic_ns"),
        "KFD acquisition completion",
    )
    deadline = _integer(kfd.get("deadline_monotonic_ns"), "KFD deadline")
    if not acquisition_started <= acquisition_completed <= deadline:
        fail("KFD acquisition timestamps differ")

    def identities(values: list[Any], label: str) -> list[tuple[int, int, int]]:
        parsed: list[tuple[int, int, int]] = []
        for item in values:
            if type(item) is not dict:
                fail(f"{label} contains a non-object identity")
            parsed.append(
                (
                    _integer(item.get("pid"), f"{label} PID", minimum=1),
                    _integer(item.get("st_dev"), f"{label} device"),
                    _integer(item.get("st_ino"), f"{label} inode", minimum=1),
                )
            )
        if parsed != sorted(set(parsed)):
            fail(f"{label} is not ascending and unique")
        return parsed

    before_identities = identities(before, "KFD before identity")
    after_identities = identities(after, "KFD after identity")
    if before_identities != after_identities:
        fail("KFD process identities changed during capture")
    positives: list[tuple[int, int]] = []
    process_identities: list[tuple[int, int, int]] = []
    for item in processes:
        if type(item) is not dict:
            fail("KFD process entry is not an object")
        pid = _integer(item.get("pid"), "KFD process PID", minimum=1)
        process_identities.append(
            (
                pid,
                _integer(item.get("st_dev"), "KFD process device"),
                _integer(item.get("st_ino"), "KFD process inode", minimum=1),
            )
        )
        vram = _integer(item.get("vram_bytes"), "KFD process VRAM")
        if vram > 0:
            positives.append((pid, vram))
    if process_identities != before_identities or positives != [(worker_pid, amd_vram)]:
        fail("AMD SMI and isolated KFD VRAM ownership differ")
    return GpuIsolationSnapshot(
        worker_pid,
        start_before,
        amd_vram,
        positives[0][1],
        tuple(pid for pid, _ in positives),
        amd_list_sha256,
        attempts,
    )


def _require_http(response: HttpResponse, url: str, body: bytes, label: str) -> None:
    if (
        type(response) is not HttpResponse
        or response.url != url
        or type(response.status) is not int
        or response.status != 200
        or type(response.body) is not bytes
        or response.body != body
    ):
        fail(f"{label} status or exact body differs")


def _http_get(http: HttpReader, url: str, body: bytes, label: str) -> HttpResponse:
    response = http.get(
        url,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        maximum_body_bytes=MAX_HTTP_BODY_BYTES,
    )
    _require_http(response, url, body, label)
    return response


def _gateway_get(
    gateway_http: GatewayNamespaceReader, container_pid: int, url: str, label: str
) -> HttpResponse:
    response = gateway_http.get(
        container_pid,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        maximum_body_bytes=MAX_HTTP_BODY_BYTES,
    )
    _require_http(response, url, GATEWAY_READY_BODY, label)
    return response


def _validate_expectation(value: OperationalExpectation) -> None:
    if (
        type(value) is not OperationalExpectation
        or SERVICE_UNIT_RE.fullmatch(value.service_unit) is None
        or type(value.gateway_pid) is not int
        or value.gateway_pid <= 0
        or type(value.worker_pid) is not int
        or value.worker_pid <= 0
        or value.gateway_pid == value.worker_pid
        or CONTAINER_NAME_RE.fullmatch(value.container_name) is None
        or CONTAINER_ID_RE.fullmatch(value.container_id) is None
        or IMAGE_ID_RE.fullmatch(value.image_id) is None
        or CONTAINER_NAME_RE.fullmatch(value.network_name) is None
        or NETWORK_ID_RE.fullmatch(value.network_id) is None
        or type(value.observer_parent_uid) is not int
        or value.observer_parent_uid < 0
        or type(value.observer_parent_gid) is not int
        or value.observer_parent_gid < 0
        or type(value.observer_parent_mode) is not int
        or value.observer_parent_mode != 0o750
        or value.gateway_ready_url != "http://172.20.0.1:8000/readyz"
    ):
        fail("operational expectation binding differs")
    _observer_path(value.observer_socket)
    for url, expected_path in (
        (value.gateway_ready_url, "/readyz"),
        (value.openwebui_health_url, "/health"),
    ):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
        ):
            fail("operational HTTP endpoint binding differs")


def _require_gpu_snapshot(value: GpuIsolationSnapshot, worker_pid: int) -> None:
    if (
        type(value) is not GpuIsolationSnapshot
        or value.worker_pid != worker_pid
        or value.worker_starttime_ticks <= 0
        or value.amd_vram_bytes <= 0
        or value.kfd_vram_bytes != value.amd_vram_bytes
        or value.positive_kfd_pids != (worker_pid,)
        or re.fullmatch(r"[0-9a-f]{64}", value.amd_list_sha256) is None
        or value.kfd_attempt_count <= 0
    ):
        fail("R9700 single-worker isolation differs")


def run_operational_preflight(
    expectation: OperationalExpectation,
    dependencies: OperationalDependencies,
) -> OperationalSnapshot:
    """Run a fully read-only, double-checked operational preflight."""

    _validate_expectation(expectation)
    observer = dependencies.observer_paths.open_path(expectation.observer_socket)
    try:
        observer_before = _require_observer_snapshot(observer.snapshot(), expectation)
        systemd_before = _capture_systemd(
            dependencies.commands,
            expectation.service_unit,
            expectation.gateway_pid,
            dependencies.monotonic_ns,
        )
        container_before = _capture_container(dependencies.commands, expectation)
        ready = _gateway_get(
            dependencies.gateway_http,
            container_before.pid,
            expectation.gateway_ready_url,
            "gateway readiness",
        )
        health = _http_get(
            dependencies.http,
            expectation.openwebui_health_url,
            OPENWEBUI_HEALTH_BODY,
            "OpenWebUI health",
        )
        gpu_before = dependencies.gpu.capture(expectation.worker_pid)
        _require_gpu_snapshot(gpu_before, expectation.worker_pid)
        _http_get(
            dependencies.http,
            expectation.openwebui_health_url,
            OPENWEBUI_HEALTH_BODY,
            "final OpenWebUI health",
        )
        _gateway_get(
            dependencies.gateway_http,
            container_before.pid,
            expectation.gateway_ready_url,
            "final gateway readiness",
        )
        gpu_after = dependencies.gpu.capture(expectation.worker_pid)
        _require_gpu_snapshot(gpu_after, expectation.worker_pid)
        container_after = _capture_container(dependencies.commands, expectation)
        systemd_after = _capture_systemd(
            dependencies.commands,
            expectation.service_unit,
            expectation.gateway_pid,
            dependencies.monotonic_ns,
        )
        observer_after = _require_observer_snapshot(observer.snapshot(), expectation)
        if systemd_after.stable_identity() != systemd_before.stable_identity():
            fail("systemd identity changed during operational preflight")
        if container_after != container_before:
            fail("OpenWebUI container identity changed during operational preflight")
        if gpu_after.stable_identity() != gpu_before.stable_identity():
            fail("R9700 worker isolation changed during operational preflight")
        if observer_after != observer_before:
            fail("observer parent identity changed during operational preflight")
        result = OperationalSnapshot(
            systemd_after,
            container_after,
            gpu_after,
            observer_after,
            ready,
            health,
        )
    except BaseException as primary:
        try:
            observer.close()
        except BaseException as cleanup:
            primary.add_note(f"observer cleanup also failed: {type(cleanup).__name__}")
        raise
    else:
        observer.close()
        return result
