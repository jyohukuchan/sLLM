#!/usr/bin/env python3
"""Production backend composition for the serial SQ8 OpenWebUI campaign."""

from __future__ import annotations

import dataclasses
import os
import re
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, cast


PYTHON_BIN = "/usr/bin/python3"
TOOLS_DIR = Path(__file__).resolve().parent
COMMAND_ENVIRONMENT = {
    "HOME": "/",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/opt/rocm/bin",
    "SYSTEMD_COLORS": "0",
    "SYSTEMD_PAGER": "",
}
MAX_STDOUT_BYTES = 64 << 10
MAX_STDERR_BYTES = 64 << 10
READ_CHUNK_BYTES = 16 << 10
PRODUCTION_IMAGE_ID = (
    "sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
PRODUCTION_NETWORK_ID = (
    "79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8"
)
IMAGE_RE = re.compile(r"(?:[a-z0-9._/-]+@)?sha256:[0-9a-f]{64}\Z")

GateName = Literal[
    "api_contract", "combined", "direct_cancel", "stop", "failure", "latency"
]
GATE_ORDER: tuple[GateName, ...] = (
    "api_contract",
    "combined",
    "direct_cancel",
    "stop",
    "failure",
    "latency",
)
GATE_SCRIPTS: Mapping[GateName, Path] = {
    "api_contract": TOOLS_DIR / "run-sq8-api-contract-gate.py",
    "combined": TOOLS_DIR / "run-openwebui-soak-gate.py",
    "direct_cancel": TOOLS_DIR / "run-sq8-direct-cancel-gate.py",
    "stop": TOOLS_DIR / "run-openwebui-stop-gate.py",
    "failure": TOOLS_DIR / "run-openwebui-failure-gate.py",
    "latency": TOOLS_DIR / "run-sq8-http-latency-gate.py",
}


class ProductionBackendError(RuntimeError):
    """A fixed production backend contract was violated."""


def fail(message: str) -> NoReturn:
    raise ProductionBackendError(message)


class SecretOwnerProtocol(Protocol):
    api_key_path: Path
    openwebui_token_path: Path

    def revalidate(self) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class SecretMasterPaths:
    """Paths owned by CampaignSecretOwner; no cleartext is retained here."""

    api_key: Path
    openwebui_token: Path
    _revalidate: Callable[[], None] = dataclasses.field(repr=False, compare=False)

    @classmethod
    def from_owner(cls, owner: SecretOwnerProtocol) -> SecretMasterPaths:
        owner.revalidate()
        for value in (owner.api_key_path, owner.openwebui_token_path):
            if not isinstance(value, Path) or not value.is_absolute():
                fail("campaign secret master path must be absolute")
        instance = object.__new__(cls)
        object.__setattr__(instance, "api_key", owner.api_key_path)
        object.__setattr__(instance, "openwebui_token", owner.openwebui_token_path)
        object.__setattr__(instance, "_revalidate", owner.revalidate)
        return instance

    def revalidate(self) -> None:
        self._revalidate()


@dataclasses.dataclass(frozen=True, slots=True)
class GateDeployment:
    http_image_id: str
    docker_network_id: str
    browser_image: str
    probe_image: str
    openwebui_url: str
    gateway_ready_url: str
    network_name: str
    service_unit: str
    expected_epoch_file: Path

    def __post_init__(self) -> None:
        if (
            self.http_image_id != PRODUCTION_IMAGE_ID
            or self.docker_network_id != PRODUCTION_NETWORK_ID
            or IMAGE_RE.fullmatch(self.browser_image) is None
            or IMAGE_RE.fullmatch(self.probe_image) is None
            or self.openwebui_url != "http://192.168.0.66:3000/"
            or self.gateway_ready_url != "http://172.20.0.1:8000/readyz"
            or self.network_name != "open-webui-network"
            or self.service_unit != "ullm-openai.service"
            or not isinstance(self.expected_epoch_file, Path)
            or not self.expected_epoch_file.is_absolute()
        ):
            fail("production gate deployment binding differs")
        if ".." in self.expected_epoch_file.parts:
            fail("restart epoch file must be absolute")


def _base(gate: GateName, output_dir: Path) -> list[str]:
    if type(gate) is not str or gate not in GATE_SCRIPTS:
        fail("unsupported production gate")
    if (
        not isinstance(output_dir, Path)
        or not output_dir.is_absolute()
        or Path(os.path.abspath(output_dir)) != output_dir
        or ".." in output_dir.parts
    ):
        fail("gate output directory must be absolute")
    return [
        PYTHON_BIN,
        "-I",
        os.fspath(GATE_SCRIPTS[gate]),
        "--output-dir",
        os.fspath(output_dir),
    ]


def build_gate_argv(
    gate: GateName,
    output_dir: Path,
    secrets: SecretMasterPaths,
    deployment: GateDeployment,
) -> tuple[str, ...]:
    """Build the only six producer commands admitted by this backend."""

    if not isinstance(secrets, SecretMasterPaths):
        fail("campaign secret master owner binding differs")
    if not isinstance(deployment, GateDeployment):
        fail("production gate deployment type differs")
    command = _base(gate, output_dir)
    if gate in {"api_contract", "direct_cancel"}:
        command += [
            "--api-key-file",
            os.fspath(secrets.api_key),
            "--http-image-id",
            deployment.http_image_id,
            "--docker-network-id",
            deployment.docker_network_id,
        ]
    elif gate == "combined":
        command += [
            "--token-file",
            os.fspath(secrets.openwebui_token),
            "--browser-image",
            deployment.browser_image,
            "--openwebui-url",
            deployment.openwebui_url,
            "--service",
            deployment.service_unit,
            "--include-smoke",
        ]
    elif gate == "stop":
        command += [
            "--token-file",
            os.fspath(secrets.openwebui_token),
            "--browser-image",
            deployment.browser_image,
            "--openwebui-url",
            deployment.openwebui_url,
            "--service",
            deployment.service_unit,
        ]
    elif gate == "failure":
        command += [
            "--token-file",
            os.fspath(secrets.openwebui_token),
            "--browser-image",
            deployment.browser_image,
            "--probe-image",
            deployment.probe_image,
            "--openwebui-url",
            deployment.openwebui_url,
            "--ready-url",
            deployment.gateway_ready_url,
            "--network",
            deployment.network_name,
            "--service",
            deployment.service_unit,
        ]
    elif gate == "latency":
        command += [
            "--api-key-file",
            os.fspath(secrets.api_key),
            "--http-image-id",
            deployment.http_image_id,
            "--docker-network-id",
            deployment.docker_network_id,
            "--expected-epoch-file",
            os.fspath(deployment.expected_epoch_file),
        ]
    else:
        fail("unsupported production gate")
    return tuple(command)


@dataclasses.dataclass(frozen=True, slots=True)
class GateCommandResult:
    stdout: bytes
    stderr: bytes


class ProcessProtocol(Protocol):
    pid: int
    returncode: int | None
    stdout: Any
    stderr: Any

    def wait(self, timeout: float | None = None) -> int: ...

    def poll(self) -> int | None: ...


ProcessFactory = Callable[..., ProcessProtocol]
KillGroup = Callable[[int, int], None]


class BoundedGateRunner:
    """Run allowlisted gate commands and always reap their process group."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 3600.0,
        process_factory: ProcessFactory | None = None,
        kill_group: KillGroup = os.killpg,
    ) -> None:
        if type(timeout_seconds) is not float or not 1.0 <= timeout_seconds <= 7200.0:
            fail("gate timeout differs from the bounded contract")
        self.timeout_seconds = timeout_seconds
        self.process_factory = (
            cast(ProcessFactory, subprocess.Popen)
            if process_factory is None
            else process_factory
        )
        self.kill_group = kill_group

    def run_gate(
        self,
        gate: GateName,
        output_dir: Path,
        secrets: SecretMasterPaths,
        deployment: GateDeployment,
    ) -> GateCommandResult:
        argv = build_gate_argv(gate, output_dir, secrets, deployment)
        return self._run_command(argv)

    def _run_command(self, command: tuple[str, ...]) -> GateCommandResult:
        argv = command
        if not _is_allowed_command(argv):
            fail("gate command is outside the fixed allowlist")
        process = self.process_factory(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=COMMAND_ENVIRONMENT,
            close_fds=True,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        try:
            if process.stdout is None or process.stderr is None:
                fail("gate process pipes are unavailable")
            selector.register(process.stdout.fileno(), selectors.EVENT_READ, "stdout")
            selector.register(process.stderr.fileno(), selectors.EVENT_READ, "stderr")
            chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
            totals = {"stdout": 0, "stderr": 0}
            limits = {"stdout": MAX_STDOUT_BYTES, "stderr": MAX_STDERR_BYTES}
            deadline = time.monotonic() + self.timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail("production gate command timed out")
                for key, _mask in selector.select(min(remaining, 0.25)):
                    chunk = os.read(key.fd, READ_CHUNK_BYTES)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    stream = str(key.data)
                    totals[stream] += len(chunk)
                    if totals[stream] > limits[stream]:
                        fail(f"gate command {stream} exceeds its byte bound")
                    chunks[stream].append(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fail("production gate command timed out")
            process.wait(timeout=remaining)
            result = GateCommandResult(
                b"".join(chunks["stdout"]), b"".join(chunks["stderr"])
            )
        except BaseException as error:
            self._terminate(process, error)
            raise
        finally:
            selector.close()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if process.returncode != 0:
            fail("production gate command failed")
        return result

    def _terminate(self, process: ProcessProtocol, primary: BaseException) -> None:
        killed = False
        try:
            self.kill_group(process.pid, signal.SIGKILL)
            killed = True
        except BaseException:
            primary.add_note(
                "gate process-group SIGKILL delivery failed; bounded reap attempted"
            )
        if killed:
            while True:
                try:
                    process.wait()
                    return
                except InterruptedError:
                    continue
                except BaseException:
                    primary.add_note("gate leader reap failed after successful SIGKILL")
                    return
        try:
            process.wait(timeout=10.0)
        except BaseException:
            primary.add_note("gate leader bounded reap failed after SIGKILL failure")


def _is_allowed_command(command: tuple[str, ...]) -> bool:
    if (
        len(command) < 5
        or any(type(item) is not str or not item or "\x00" in item for item in command)
        or command[:2] != (PYTHON_BIN, "-I")
    ):
        return False
    try:
        script = Path(command[2])
    except TypeError:
        return False
    scripts = {path: gate for gate, path in GATE_SCRIPTS.items()}
    gate = scripts.get(script)
    if gate is None or command[3] != "--output-dir":
        return False
    layouts: Mapping[GateName, tuple[str, ...]] = {
        "api_contract": (
            "--output-dir",
            "--api-key-file",
            "--http-image-id",
            "--docker-network-id",
        ),
        "combined": (
            "--output-dir",
            "--token-file",
            "--browser-image",
            "--openwebui-url",
            "--service",
            "--include-smoke",
        ),
        "direct_cancel": (
            "--output-dir",
            "--api-key-file",
            "--http-image-id",
            "--docker-network-id",
        ),
        "stop": (
            "--output-dir",
            "--token-file",
            "--browser-image",
            "--openwebui-url",
            "--service",
        ),
        "failure": (
            "--output-dir",
            "--token-file",
            "--browser-image",
            "--probe-image",
            "--openwebui-url",
            "--ready-url",
            "--network",
            "--service",
        ),
        "latency": (
            "--output-dir",
            "--api-key-file",
            "--http-image-id",
            "--docker-network-id",
            "--expected-epoch-file",
        ),
    }
    index = 3
    for option in layouts[gate]:
        if index >= len(command) or command[index] != option:
            return False
        index += 1
        if option == "--include-smoke":
            continue
        if (
            index >= len(command)
            or not command[index]
            or command[index].startswith("--")
        ):
            return False
        index += 1
    return index == len(command)


class OwnerProtocol(Protocol):
    def close(self) -> None: ...


class BackendBridge(Protocol):
    """SystemRuntime/RuntimeSnapshots and campaign artifact factory boundary."""

    def now_ns(self) -> int: ...
    def scan_evidence(self, raw: bytes, label: str) -> None: ...
    def make_session_writer(self, path: Path) -> Any: ...
    def make_resource_writer(self, path: Path) -> Any: ...
    def make_journal_capture(
        self, path: Path, boot_id: str, normal_epoch: Any
    ) -> Any: ...
    def preflight(self, work_dir: Path) -> Any: ...
    def make_resource_adapter(self, **arguments: Any) -> Any: ...
    def final(self, work_dir: Path) -> Any: ...
    def close(self) -> None: ...


class GateBindingsFactory(Protocol):
    def build(self, gate: GateName, work_dir: Path) -> Any: ...


class GateRunner(Protocol):
    def run_gate(
        self,
        gate: GateName,
        output_dir: Path,
        secrets: SecretMasterPaths,
        deployment: GateDeployment,
    ) -> GateCommandResult: ...


GateIngestor = Callable[[Path, Any], Any]


@dataclasses.dataclass(frozen=True, slots=True)
class GateIngestors:
    api_contract: GateIngestor
    combined: GateIngestor
    direct_cancel: GateIngestor
    stop: GateIngestor
    failure: GateIngestor
    latency: GateIngestor

    def for_gate(self, gate: GateName) -> GateIngestor:
        return cast(GateIngestor, getattr(self, gate))


class ResourceAdapter:
    """Close a collector and any subordinate owners in reverse creation order."""

    def __init__(self, collector: Any, owners: Sequence[OwnerProtocol] = ()) -> None:
        self.collector = collector
        self.owners = tuple(owners)
        self.closed = False

    def collect_normal(self, *, expected_identity: Any | None = None) -> Any:
        return self.collector.collect_normal(expected_identity=expected_identity)

    def collect_restart(
        self, normal_identity: Any, *, expected_identity: Any | None = None
    ) -> Any:
        return self.collector.collect_restart(
            normal_identity, expected_identity=expected_identity
        )

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for owner in reversed(self.owners):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            raise errors[0]


class ProductionCampaignBackend:
    """Run each producer once, then return only independently ingested evidence."""

    def __init__(
        self,
        *,
        bridge: BackendBridge,
        runner: GateRunner,
        secrets: SecretMasterPaths,
        deployment: GateDeployment,
        bindings: GateBindingsFactory,
        ingestors: GateIngestors,
        owners: Sequence[OwnerProtocol] = (),
    ) -> None:
        self.bridge = bridge
        self.runner = runner
        self.secrets = secrets
        self.deployment = deployment
        self.bindings = bindings
        self.ingestors = ingestors
        self.owners = tuple(owners)
        self.completed: list[GateName] = []
        self.attempted: list[GateName] = []
        self.poisoned = False
        self.closed = False

    def now_ns(self) -> int:
        return self.bridge.now_ns()

    def scan_evidence(self, raw: bytes, label: str) -> None:
        self.bridge.scan_evidence(raw, label)

    def make_session_writer(self, path: Path) -> Any:
        return self.bridge.make_session_writer(path)

    def make_resource_writer(self, path: Path) -> Any:
        return self.bridge.make_resource_writer(path)

    def make_journal_capture(self, path: Path, boot_id: str, normal_epoch: Any) -> Any:
        return self.bridge.make_journal_capture(path, boot_id, normal_epoch)

    def preflight(self, work_dir: Path) -> Any:
        return self.bridge.preflight(work_dir)

    def api_contract(self, work_dir: Path) -> Any:
        return self._run_gate("api_contract", work_dir)

    def combined(self, work_dir: Path) -> Any:
        return self._run_gate("combined", work_dir)

    def direct_cancel(self, work_dir: Path) -> Any:
        return self._run_gate("direct_cancel", work_dir)

    def stop(self, work_dir: Path) -> Any:
        return self._run_gate("stop", work_dir)

    def failure(self, work_dir: Path) -> Any:
        return self._run_gate("failure", work_dir)

    def latency(self, work_dir: Path) -> Any:
        return self._run_gate("latency", work_dir)

    def _run_gate(self, gate: GateName, work_dir: Path) -> Any:
        expected = (
            GATE_ORDER[len(self.attempted)]
            if len(self.attempted) < len(GATE_ORDER)
            else None
        )
        if self.closed or self.poisoned or gate != expected:
            fail("production gate order differs")
        self.attempted.append(gate)
        try:
            self.secrets.revalidate()
            self.runner.run_gate(gate, work_dir, self.secrets, self.deployment)
            result = self.ingestors.for_gate(gate)(
                work_dir, self.bindings.build(gate, work_dir)
            )
        except BaseException:
            self.poisoned = True
            raise
        self.completed.append(gate)
        return result

    def make_resource_adapter(self, **arguments: Any) -> Any:
        return self.bridge.make_resource_adapter(**arguments)

    def final(self, work_dir: Path) -> Any:
        if tuple(self.completed) != GATE_ORDER:
            fail("production finalization began before every gate completed")
        return self.bridge.final(work_dir)

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for owner in (self.bridge, *reversed(self.owners)):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            raise errors[0]


__all__ = [
    "BoundedGateRunner",
    "GateCommandResult",
    "GateDeployment",
    "GateIngestors",
    "GATE_ORDER",
    "ProductionBackendError",
    "ProductionCampaignBackend",
    "ResourceAdapter",
    "SecretMasterPaths",
    "build_gate_argv",
]
