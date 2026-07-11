#!/usr/bin/env python3
"""Production backend composition for the serial SQ8 OpenWebUI campaign."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
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
PRODUCTION_BROWSER_IMAGE_ID = (
    "sha256:dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa"
)
PRODUCTION_PROBE_IMAGE_ID = PRODUCTION_IMAGE_ID
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
            or self.browser_image != PRODUCTION_BROWSER_IMAGE_ID
            or self.probe_image != PRODUCTION_PROBE_IMAGE_ID
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


class ApiSecretUser(Protocol):
    def use_api_secret(self, callback: Callable[[bytes], Any]) -> Any: ...


class GitAnchorProtocol(Protocol):
    commit: str
    status_raw: bytes

    def revalidate(self) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class SystemBridgeInputs:
    identity: Any
    resource: Any
    git_anchor: GitAnchorProtocol
    secret_owner: ApiSecretUser
    trusted_http_client_source: bytes
    trusted_http_client_sha256: str
    repo_root: Path
    amd_smi: str


@dataclasses.dataclass(frozen=True, slots=True)
class SystemBridgeFactories:
    secret_guard: Callable[[bytes], Any]
    runtime_snapshots: Callable[[bytes, bytes], Any]
    runtime_config: Callable[[Any, str], Any]
    system_runtime: Callable[[Any, Path, Any, Any, bool], Any]
    session_writer: Callable[[Path, Any], Any]
    resource_writer: Callable[[Path, Any], Any]
    journal_capture: Callable[..., Any]
    resource_claims: Callable[[Any, Any, Callable[[], int], Callable[[int], None]], Any]
    resource_collector: Callable[..., Any]
    preflight_result: Callable[[bytes, bytes, dict[str, Any], dict[str, Any]], Any]
    final_result: Callable[[dict[str, Any], str, int, str, str], Any]


def system_bridge_factories(
    collector: Any, campaign: Any, orchestrator: Any
) -> SystemBridgeFactories:
    """Bind the bridge to the existing collector, journal, and result classes."""

    return SystemBridgeFactories(
        secret_guard=collector.SecretGuard,
        runtime_snapshots=collector.RuntimeSnapshots.create,
        runtime_config=lambda identities, amd_smi: (
            collector.SystemRuntimeConfig.for_full_campaign(
                identities=identities, amd_smi=amd_smi
            )
        ),
        system_runtime=lambda config, root, guard, snapshots, capture_journal: (
            collector.SystemRuntime(
                config,
                root,
                guard,
                snapshots,
                capture_journal=capture_journal,
            )
        ),
        session_writer=collector.SessionWriter,
        resource_writer=collector.AtomicJsonlWriter,
        journal_capture=campaign.CampaignJournalCapture,
        resource_claims=collector.CampaignResourceLifecycleClaims,
        resource_collector=collector.ResourceSegmentCollector,
        preflight_result=orchestrator.PreflightPhaseResult,
        final_result=orchestrator.FinalPhaseResult,
    )


class SystemCampaignBridge:
    """Compose the existing collector runtime without retaining cleartext itself."""

    MAX_HTTP_CLIENT_SOURCE_BYTES = 1 << 20

    def __init__(
        self,
        inputs: SystemBridgeInputs,
        factories: SystemBridgeFactories,
        *,
        scan_evidence: Callable[[bytes, str], None],
    ) -> None:
        raw = inputs.trusted_http_client_source
        if (
            type(raw) is not bytes
            or not raw
            or len(raw) > self.MAX_HTTP_CLIENT_SOURCE_BYTES
            or hashlib.sha256(raw).hexdigest() != inputs.trusted_http_client_sha256
            or not inputs.repo_root.is_absolute()
        ):
            fail("trusted HTTP client source binding differs")
        self.inputs = inputs
        self.factories = factories
        self._scan = scan_evidence
        self.guard: Any | None = None
        self.snapshots: Any | None = None
        self.runtime: Any | None = None
        self.started = False
        self.closed = False
        self._initialization_attempted = False
        self._initialization_succeeded = False

    def _initialize_once(self) -> None:
        if self.closed:
            fail("system campaign bridge is closed")
        if self._initialization_attempted:
            if self._initialization_succeeded:
                return
            fail("system campaign bridge initialization already failed")
        self._initialization_attempted = True
        callback_count = 0

        def create(secret: bytes) -> None:
            nonlocal callback_count
            if callback_count != 0:
                fail("API secret owner invoked its consumer more than once")
            callback_count += 1
            guard = factories.secret_guard(secret)
            self.guard = guard
            snapshots = factories.runtime_snapshots(raw, secret)
            self.snapshots = snapshots
            header_identities = inputs.resource.session_header_fields["identities"]
            runtime_identity_keys = (
                "openwebui",
                "docker_network_id",
                "gateway_source_sha256",
                "worker_source_sha256",
                "worker_binary_sha256",
            )
            if type(header_identities) is not dict or any(
                key not in header_identities for key in runtime_identity_keys
            ):
                fail("system runtime identities are incomplete")
            identities = {key: header_identities[key] for key in runtime_identity_keys}
            config = factories.runtime_config(identities, inputs.amd_smi)
            runtime = factories.system_runtime(
                config, inputs.repo_root, guard, snapshots, False
            )
            self.runtime = runtime

        inputs = self.inputs
        factories = self.factories
        raw = inputs.trusted_http_client_source
        inputs.secret_owner.use_api_secret(create)
        if (
            callback_count != 1
            or self.guard is None
            or self.snapshots is None
            or self.runtime is None
        ):
            fail("API secret owner did not create the campaign runtime")
        self._initialization_succeeded = True

    def _require_open(self) -> tuple[Any, Any]:
        self._initialize_once()
        if self.closed or self.guard is None or self.runtime is None:
            fail("system campaign bridge is closed")
        return self.guard, self.runtime

    def _start(self) -> Any:
        _guard, runtime = self._require_open()
        if not self.started:
            runtime.start()
            self.started = True
        return runtime

    def now_ns(self) -> int:
        _guard, runtime = self._require_open()
        return int(runtime.now_ns())

    def scan_evidence(self, raw: bytes, label: str) -> None:
        if self.closed or self.guard is None:
            self._scan(raw, label)
            return
        guard, _runtime = self._require_open()
        guard.reject(raw, label)
        self._scan(raw, label)

    def make_session_writer(self, path: Path) -> Any:
        guard, _runtime = self._require_open()
        return self.factories.session_writer(path, guard)

    def make_resource_writer(self, path: Path) -> Any:
        guard, _runtime = self._require_open()
        return self.factories.resource_writer(path, guard)

    def make_journal_capture(self, path: Path, boot_id: str, normal_epoch: Any) -> Any:
        self._require_open()
        return self.factories.journal_capture(
            path, boot_id, normal_epoch, scan_raw=self.scan_evidence
        )

    def preflight(self, work_dir: Path) -> Any:
        del work_dir
        self.inputs.git_anchor.revalidate()
        self._start()
        artifacts = self.inputs.identity.identity_artifacts
        contract = self.inputs.resource
        return self.factories.preflight_result(
            artifacts.environment_bytes,
            artifacts.model_identity_bytes,
            contract.session_header_fields,
            contract.resource_header,
        )

    def make_resource_adapter(self, **arguments: Any) -> ResourceAdapter:
        guard, runtime = self._require_open()
        journal = arguments["journal"]
        claims = self.factories.resource_claims(
            journal, arguments["session"], runtime.now_ns, runtime.wait_until
        )
        collector = self.factories.resource_collector(
            self.inputs.resource.segment_config,
            arguments["stage_path"],
            guard,
            runtime,
            arguments["session"],
            arguments["resource"],
            claims,
        )
        return ResourceAdapter(collector)

    def final(self, work_dir: Path) -> Any:
        del work_dir
        _guard, runtime = self._require_open()
        probe = runtime.lifecycle_probe()
        self.inputs.git_anchor.revalidate()
        status = self.inputs.git_anchor.status_raw.decode("utf-8", errors="strict")
        identity_fields = dataclasses.asdict(probe.identity)
        record = {
            "record_type": "lifecycle_probe",
            "phase": "final",
            "case_id": "final-service-ready",
            "fields": {
                "probe": "final-service-ready",
                "observed_monotonic_ns": probe.observed_monotonic_ns,
                "service_active": probe.service_active,
                "ready_http_status": probe.ready_http_status,
                **identity_fields,
            },
        }
        completed_ns = int(runtime.now_ns())
        completed_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.factories.final_result(
            record,
            completed_utc,
            completed_ns,
            self.inputs.git_anchor.commit,
            status,
        )

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for attribute in ("runtime", "snapshots"):
            owner = getattr(self, attribute)
            if owner is None:
                continue
            try:
                owner.close()
                setattr(self, attribute, None)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self.guard = None
        self.closed = True


@dataclasses.dataclass(frozen=True, slots=True)
class GateBindingContext:
    gate: GateName
    work_dir: Path
    normal_identity: Any
    restart_identity: Any | None
    restart_epoch_file: Path | None
    restart_epoch_sha256: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class ProductionBindingInputs:
    environment: Mapping[str, Any]
    expected_source_role_paths: Mapping[str, str]
    repo_root: Path
    binding_types: Mapping[GateName, Callable[..., Any]]
    normal_identity: Any
    restart_identity: Callable[[], Any]
    secret_guard: Any
    forbidden_values: tuple[bytes, ...]
    http_image_id: str
    docker_network_id: str
    docker_network_subnet: str
    docker_network_gateway: str
    browser_image_reference: str
    browser_image_content_id: str
    probe_image_reference: str
    probe_image_content_digest: str
    openwebui_url: str
    service_unit: str
    service_user: str
    boot_id: str
    uid: int
    gid: int


def _source_bindings(inputs: ProductionBindingInputs) -> dict[str, tuple[Path, str]]:
    sources = inputs.environment.get("sources")
    expected = inputs.expected_source_role_paths
    if (
        type(sources) is not list
        or len(sources) != 70
        or len(expected) != 70
        or not inputs.repo_root.is_absolute()
    ):
        fail("production binding source set differs")
    result: dict[str, tuple[Path, str]] = {}
    seen_paths: set[str] = set()
    for item in sources:
        if type(item) is not dict or set(item) != {"role", "path", "bytes", "sha256"}:
            fail("production binding source record differs")
        role = item["role"]
        relative = item["path"]
        digest = item["sha256"]
        if (
            type(role) is not str
            or role in result
            or expected.get(role) != relative
            or type(relative) is not str
            or relative in seen_paths
            or type(item["bytes"]) is not int
            or item["bytes"] < 1
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            fail("production binding source identity differs")
        path = inputs.repo_root / relative
        if not path.is_absolute() or ".." in Path(relative).parts:
            fail("production binding source path differs")
        result[role] = (path, digest)
        seen_paths.add(relative)
    if set(result) != set(expected):
        fail("production binding source role coverage differs")
    return result


def _identity_values(identity: Any) -> dict[str, Any]:
    names = (
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    )
    values = {name: getattr(identity, name, None) for name in names}
    if type(values["control_group"]) is not str or any(
        type(values[name]) is not int for name in names[1:]
    ):
        fail("production binding process identity differs")
    return values


def _source_fields(
    sources: Mapping[str, tuple[Path, str]], mapping: Mapping[str, str]
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for prefix, role in mapping.items():
        try:
            path, digest = sources[role]
        except KeyError:
            fail("production binding required source is missing")
        fields[f"{prefix}_source"] = path
        fields[f"{prefix}_source_sha256"] = digest
    return fields


def _instantiate_binding(
    inputs: ProductionBindingInputs, gate: GateName, values: dict[str, Any]
) -> Any:
    if set(values) != set(EXPECTED_BINDING_FIELDS[gate]):
        fail("production ingestor binding field set differs")
    try:
        result = inputs.binding_types[gate](**values)
    except (KeyError, TypeError, ValueError):
        fail("production ingestor binding construction failed")
    if (
        tuple(field.name for field in dataclasses.fields(result))
        != EXPECTED_BINDING_FIELDS[gate]
    ):
        fail("production ingestor binding type differs")
    return result


def production_binding_builders(
    inputs: ProductionBindingInputs,
) -> Mapping[GateName, Callable[[GateBindingContext], Any]]:
    """Build all six exact independent-ingestor dataclasses from prepared pins."""

    if (
        inputs.http_image_id != PRODUCTION_IMAGE_ID
        or inputs.docker_network_id != PRODUCTION_NETWORK_ID
        or inputs.docker_network_subnet != "172.20.0.0/16"
        or inputs.docker_network_gateway != "172.20.0.1"
        or inputs.browser_image_reference != PRODUCTION_BROWSER_IMAGE_ID
        or inputs.browser_image_content_id != PRODUCTION_BROWSER_IMAGE_ID
        or inputs.probe_image_reference != PRODUCTION_PROBE_IMAGE_ID
        or inputs.probe_image_content_digest != PRODUCTION_PROBE_IMAGE_ID
        or inputs.openwebui_url != "http://192.168.0.66:3000/"
        or inputs.service_unit != "ullm-openai.service"
        or inputs.service_user != "homelab1"
        or re.fullmatch(r"[0-9a-f]{32}", inputs.boot_id) is None
        or inputs.uid != 1000
        or inputs.gid != 1000
        or not callable(getattr(inputs.secret_guard, "reject", None))
    ):
        fail("production binding deployment snapshot differs")
    sources = _source_bindings(inputs)
    normal = _identity_values(inputs.normal_identity)
    common = {
        "service_unit": inputs.service_unit,
        "service_user": inputs.service_user,
        "boot_id": inputs.boot_id,
        "control_group": normal["control_group"],
        "gateway_pid": normal["gateway_pid"],
        "gateway_starttime_ticks": normal["gateway_starttime_ticks"],
        "worker_pid": normal["worker_pid"],
        "worker_starttime_ticks": normal["worker_starttime_ticks"],
        "restart_count": normal["n_restarts"],
        "uid": inputs.uid,
        "gid": inputs.gid,
        "forbidden_values": inputs.forbidden_values,
    }

    def build(context: GateBindingContext) -> Any:
        gate = context.gate
        if gate == "api_contract":
            values = {
                **_source_fields(
                    sources,
                    {
                        "gate": "gate_api_contract",
                        "direct": "gate_direct_cancel",
                        "collector": "release_collector",
                        "http_client": "http_client",
                        "gateway_app": "gateway_app",
                        "gateway_errors": "gateway_errors",
                        "gateway_schemas": "gateway_schemas",
                    },
                ),
                "http_image_id": inputs.http_image_id,
                "docker_network_id": inputs.docker_network_id,
                **common,
            }
        elif gate == "combined":
            values = {
                **_source_fields(
                    sources,
                    {
                        "gate": "gate_openwebui_soak",
                        "support": "gate_openwebui_stop",
                    },
                ),
                "browser_script": sources["browser_soak"][0],
                "browser_script_sha256": sources["browser_soak"][1],
                "browser_image_reference": inputs.browser_image_reference,
                "browser_image_content_id": inputs.browser_image_content_id,
                "openwebui_base_url": inputs.openwebui_url,
                "service_unit": inputs.service_unit,
                "boot_id": inputs.boot_id,
                "gateway_pid": normal["gateway_pid"],
                "uid": inputs.uid,
                "gid": inputs.gid,
                "restart_count": normal["n_restarts"],
                "forbidden_values": inputs.forbidden_values,
            }
        elif gate == "direct_cancel":
            values = {
                **_source_fields(
                    sources,
                    {
                        "gate": "gate_direct_cancel",
                        "collector": "release_collector",
                        "http_client": "http_client",
                    },
                ),
                "http_image_id": inputs.http_image_id,
                "docker_network_id": inputs.docker_network_id,
                **common,
            }
        elif gate == "stop":
            values = {
                **_source_fields(
                    sources,
                    {"gate": "gate_openwebui_stop"},
                ),
                "browser_script": sources["browser_stop"][0],
                "browser_script_sha256": sources["browser_stop"][1],
                "browser_image_reference": inputs.browser_image_reference,
                "browser_image_content_id": inputs.browser_image_content_id,
                "openwebui_url": inputs.openwebui_url,
                **common,
            }
        elif gate == "failure":
            restart = _identity_values(inputs.restart_identity())
            values = {
                **_source_fields(
                    sources,
                    {
                        "gate": "gate_openwebui_failure",
                        "hook": "gate_openwebui_failure_hook",
                        "browser": "browser_failure",
                    },
                ),
                "browser_image_reference": inputs.browser_image_reference,
                "browser_image_content_digest": inputs.browser_image_content_id,
                "probe_image_reference": inputs.probe_image_reference,
                "probe_image_content_digest": inputs.probe_image_content_digest,
                "docker_network_id": inputs.docker_network_id,
                "docker_network_subnet": inputs.docker_network_subnet,
                "docker_network_gateway": inputs.docker_network_gateway,
                "service_unit": inputs.service_unit,
                "service_user": inputs.service_user,
                "boot_id": inputs.boot_id,
                "control_group": normal["control_group"],
                "normal_gateway_pid": normal["gateway_pid"],
                "normal_gateway_starttime_ticks": normal["gateway_starttime_ticks"],
                "normal_worker_pid": normal["worker_pid"],
                "normal_worker_starttime_ticks": normal["worker_starttime_ticks"],
                "normal_restart_count": normal["n_restarts"],
                "restart_gateway_pid": restart["gateway_pid"],
                "restart_gateway_starttime_ticks": restart["gateway_starttime_ticks"],
                "restart_worker_pid": restart["worker_pid"],
                "restart_worker_starttime_ticks": restart["worker_starttime_ticks"],
                "restart_restart_count": restart["n_restarts"],
                "uid": inputs.uid,
                "gid": inputs.gid,
                "forbidden_values": inputs.forbidden_values,
            }
        else:
            restart = _identity_values(context.restart_identity)
            values = {
                **_source_fields(
                    sources,
                    {
                        "gate": "gate_http_latency",
                        "direct": "gate_direct_cancel",
                        "collector": "release_collector",
                        "http_client": "http_client",
                    },
                ),
                "restart_epoch_file": context.restart_epoch_file,
                "restart_epoch_sha256": context.restart_epoch_sha256,
                "http_image_id": inputs.http_image_id,
                "docker_network_id": inputs.docker_network_id,
                "service_unit": inputs.service_unit,
                "service_user": inputs.service_user,
                "boot_id": inputs.boot_id,
                "control_group": restart["control_group"],
                "gateway_pid": restart["gateway_pid"],
                "gateway_starttime_ticks": restart["gateway_starttime_ticks"],
                "worker_pid": restart["worker_pid"],
                "worker_starttime_ticks": restart["worker_starttime_ticks"],
                "restart_count": restart["n_restarts"],
                "uid": inputs.uid,
                "gid": inputs.gid,
                "forbidden_values": inputs.forbidden_values,
            }
        return _instantiate_binding(inputs, gate, values)

    return {gate: build for gate in GATE_ORDER}


class ProductionGateBindingsFactory:
    """Build exact ingestor bindings and pin the one post-failure epoch."""

    def __init__(
        self,
        normal_identity: Any,
        builders: Mapping[GateName, Callable[[GateBindingContext], Any]],
        restart_extractor: Callable[[Any, Path], tuple[Any, "RestartEpochOwner"]],
    ) -> None:
        if set(builders) != set(GATE_ORDER):
            fail("production gate binding builder set differs")
        self.normal_identity = normal_identity
        self.builders = dict(builders)
        self.restart_extractor = restart_extractor
        self.restart_identity: Any | None = None
        self.restart_epoch_file: Path | None = None
        self.restart_epoch_sha256: str | None = None
        self.restart_epoch_owner: RestartEpochOwner | None = None

    def confirm_restart(
        self, identity: Any, epoch_file: Path, epoch_sha256: str
    ) -> None:
        if (
            self.restart_identity is not None
            or not epoch_file.is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", epoch_sha256) is None
        ):
            fail("production restart epoch binding differs")
        self.restart_identity = identity
        self.restart_epoch_file = epoch_file
        self.restart_epoch_sha256 = epoch_sha256

    def confirm_failure(self, result: Any, work_dir: Path) -> Path:
        identity, owner = self.restart_extractor(result, work_dir)
        if not isinstance(owner, RestartEpochOwner):
            fail("production restart epoch owner type differs")
        self.restart_epoch_owner = owner
        try:
            owner.revalidate()
            self.confirm_restart(identity, owner.path, owner.sha256)
        except BaseException as error:
            cleanup_failed = False
            try:
                owner.close()
            except BaseException:
                cleanup_failed = True
                error.add_note("restart epoch owner cleanup also failed")
            if not cleanup_failed:
                self.restart_epoch_owner = None
            raise
        return owner.path

    def build(self, gate: GateName, work_dir: Path) -> Any:
        if gate == "latency" and self.restart_identity is None:
            fail("latency binding lacks the confirmed restart epoch")
        if gate == "latency":
            if self.restart_epoch_owner is None:
                fail("latency binding lacks the restart epoch owner")
            self.restart_epoch_owner.revalidate()
        return self.builders[gate](
            GateBindingContext(
                gate,
                work_dir,
                self.normal_identity,
                self.restart_identity,
                self.restart_epoch_file,
                self.restart_epoch_sha256,
            )
        )

    def confirm_latency(self) -> None:
        if self.restart_epoch_owner is None:
            fail("latency completion lacks the restart epoch owner")
        self.restart_epoch_owner.revalidate()
        self.restart_epoch_owner.close()
        self.restart_epoch_owner = None

    def close(self) -> None:
        if self.restart_epoch_owner is None:
            return
        self.restart_epoch_owner.close()
        self.restart_epoch_owner = None


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclasses.dataclass(slots=True)
class RestartEpochOwner:
    path: Path
    sha256: str
    identity: Mapping[str, Any]
    directory_fd: int
    file_fd: int
    directory_identity: tuple[int, ...]
    file_identity: tuple[int, ...]
    closed: bool = False

    def revalidate(self) -> None:
        if self.closed or self.directory_fd < 0 or self.file_fd < 0:
            fail("restart epoch owner is closed")
        try:
            directory = os.fstat(self.directory_fd)
            directory_entry = os.stat(self.path.parent, follow_symlinks=False)
            file_value = os.fstat(self.file_fd)
            file_entry = os.stat(
                self.path.name, dir_fd=self.directory_fd, follow_symlinks=False
            )
        except OSError:
            fail("restart epoch owner entry is unavailable")
        if (
            _stat_identity(directory) != self.directory_identity
            or _stat_identity(directory_entry) != self.directory_identity
            or _stat_identity(file_value) != self.file_identity
            or _stat_identity(file_entry) != self.file_identity
        ):
            fail("restart epoch owner identity changed")

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for attribute in ("file_fd", "directory_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
                setattr(self, attribute, -1)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self.closed = True


def write_restart_epoch(
    stage_path: Path,
    service_identity: Mapping[str, Any],
    secret_guard: Any,
) -> RestartEpochOwner:
    """Atomically create the one private latency epoch pin after failure."""

    fields = (
        "unit",
        "user",
        "uid",
        "gid",
        "control_group",
        "boot_id",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    )
    integer_fields = (
        "uid",
        "gid",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    )
    if (
        set(service_identity) != set(fields)
        or not stage_path.is_absolute()
        or Path(os.path.abspath(stage_path)) != stage_path
        or ".." in stage_path.parts
        or os.geteuid() != 1000
        or os.getegid() != 1000
        or service_identity.get("unit") != "ullm-openai.service"
        or service_identity.get("user") != "homelab1"
        or service_identity.get("uid") != 1000
        or service_identity.get("gid") != 1000
        or type(service_identity.get("boot_id")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", service_identity["boot_id"]) is None
        or type(service_identity.get("control_group")) is not str
        or not service_identity["control_group"].startswith("/")
        or len(service_identity["control_group"]) > 4096
        or any(
            type(service_identity.get(name)) is not int
            or service_identity.get(name, -1) < 0
            for name in integer_fields
        )
        or service_identity.get("gateway_pid", 0) < 1
        or service_identity.get("worker_pid", 0) < 1
        or service_identity.get("gateway_starttime_ticks", 0) < 1
        or service_identity.get("worker_starttime_ticks", 0) < 1
        or service_identity.get("uid", 0) > (1 << 32) - 1
        or service_identity.get("gid", 0) > (1 << 32) - 1
        or service_identity.get("gateway_pid", 0) > (1 << 31) - 1
        or service_identity.get("worker_pid", 0) > (1 << 31) - 1
        or service_identity.get("gateway_starttime_ticks", 0) > (1 << 63) - 1
        or service_identity.get("worker_starttime_ticks", 0) > (1 << 63) - 1
        or service_identity.get("n_restarts", 0) > (1 << 31) - 1
    ):
        fail("restart epoch service identity differs")
    try:
        for name, limit in (
            ("unit", 255),
            ("user", 64),
            ("boot_id", 32),
            ("control_group", 4096),
        ):
            encoded = service_identity[name].encode("ascii", errors="strict")
            if not encoded or len(encoded) > limit or b"\x00" in encoded:
                fail("restart epoch service string differs")
    except UnicodeError:
        fail("restart epoch service string is not ASCII")
    value = {
        "schema_version": "ullm.sq8.resource_restart_epoch.v1",
        "phase": "resource_restart",
        "service_identity": dict(service_identity),
    }
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("restart epoch cannot be encoded")
    if len(raw) > 16 << 10:
        fail("restart epoch exceeds its byte bound")
    secret_guard.reject(raw, "restart epoch")
    name = "resource-restart-epoch.json"
    path = stage_path / name
    directory_fd = -1
    descriptor = -1
    created = False
    file_anchor: tuple[int, int] | None = None
    try:
        directory_fd = os.open(
            stage_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        before = os.fstat(directory_fd)
        before_entry = os.stat(stage_path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before_entry.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_uid != os.geteuid()
            or before.st_gid != os.getegid()
            or (before.st_dev, before.st_ino)
            != (before_entry.st_dev, before_entry.st_ino)
        ):
            fail("restart epoch stage is not a directory")
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        opened = os.fstat(descriptor)
        file_anchor = (opened.st_dev, opened.st_ino)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset : offset + 4096])
            if written <= 0:
                fail("restart epoch write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        file_identity = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        after = os.fstat(directory_fd)
        after_entry = os.stat(stage_path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (after_entry.st_dev, after_entry.st_ino)
            or (file_identity.st_dev, file_identity.st_ino)
            != (entry.st_dev, entry.st_ino)
            or not stat.S_ISREG(file_identity.st_mode)
            or stat.S_IMODE(file_identity.st_mode) != 0o600
            or file_identity.st_nlink != 1
            or file_identity.st_size != len(raw)
        ):
            fail("restart epoch identity changed")
        os.fsync(directory_fd)
        return RestartEpochOwner(
            path,
            hashlib.sha256(raw).hexdigest(),
            dict(service_identity),
            directory_fd,
            descriptor,
            _stat_identity(after),
            _stat_identity(file_identity),
        )
    except BaseException as error:
        if created and directory_fd >= 0:
            try:
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if file_anchor == (entry.st_dev, entry.st_ino):
                    os.unlink(name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            except BaseException:
                error.add_note("restart epoch partial-file cleanup also failed")
        for descriptor_value, note in (
            (descriptor, "restart epoch file descriptor cleanup also failed"),
            (directory_fd, "restart epoch directory descriptor cleanup also failed"),
        ):
            if descriptor_value < 0:
                continue
            try:
                os.close(descriptor_value)
            except BaseException:
                error.add_note(note)
        raise


def production_binding_factory(
    inputs: ProductionBindingInputs,
) -> ProductionGateBindingsFactory:
    """Create bindings plus the canonical one-shot restart epoch writer."""

    builders = production_binding_builders(inputs)

    def extract(result: Any, work_dir: Path) -> tuple[Any, RestartEpochOwner]:
        restart = inputs.restart_identity()
        identity = _identity_values(restart)
        record = getattr(result, "restart_probe_record", None)
        fields = record.get("fields") if type(record) is dict else None
        if type(fields) is not dict or any(
            fields.get(name) != identity[name]
            for name in (
                "control_group",
                "gateway_pid",
                "gateway_starttime_ticks",
                "worker_pid",
                "worker_starttime_ticks",
                "n_restarts",
            )
        ):
            fail("failure result restart identity differs")
        service = {
            "unit": inputs.service_unit,
            "user": inputs.service_user,
            "uid": inputs.uid,
            "gid": inputs.gid,
            "control_group": identity["control_group"],
            "boot_id": inputs.boot_id,
            "gateway_pid": identity["gateway_pid"],
            "gateway_starttime_ticks": identity["gateway_starttime_ticks"],
            "worker_pid": identity["worker_pid"],
            "worker_starttime_ticks": identity["worker_starttime_ticks"],
            "n_restarts": identity["n_restarts"],
        }
        evidence = write_restart_epoch(work_dir, service, inputs.secret_guard)
        return restart, evidence

    return ProductionGateBindingsFactory(inputs.normal_identity, builders, extract)


@dataclasses.dataclass(frozen=True, slots=True)
class IngestorModules:
    api_contract: Any
    openwebui: Any
    stop: Any
    failure: Any
    latency: Any


def production_gate_ingestors(modules: IngestorModules) -> GateIngestors:
    """Adapt the six existing independent bundle ingestors without trusting producers."""

    return GateIngestors(
        modules.api_contract.ingest_api_contract_bundle,
        modules.openwebui.ingest_combined_soak_bundle,
        modules.openwebui.ingest_direct_cancel_bundle,
        modules.stop.ingest_stop_gate_bundle,
        modules.failure.ingest_failure_gate_bundle,
        modules.latency.ingest_latency_gate_bundle,
    )


def execute_prepared_campaign(prepared: Any, orchestrator: Any) -> Path:
    """Build every fixed production adapter and publish one prepared campaign."""

    import sq8_api_contract_gate_ingest as api_ingest
    import sq8_full_campaign_renderer as renderer_module
    import sq8_http_latency_gate_ingest as latency_ingest
    import sq8_openwebui_campaign as campaign_module
    import sq8_openwebui_failure_gate_ingest as failure_ingest
    import sq8_openwebui_gate_ingest as openwebui_ingest
    import sq8_openwebui_stop_gate_ingest as stop_ingest

    runtime = prepared.runtime
    collector = runtime.collector
    production = runtime.production_module
    identity_module = runtime.identity_module
    environment = prepared.identity.identity_artifacts.environment
    source_items = environment.get("sources")
    if type(source_items) is not list:
        fail("prepared production sources differ")
    http_sources = [
        item
        for item in source_items
        if type(item) is dict and item.get("role") == "http_client"
    ]
    if len(http_sources) != 1:
        fail("prepared HTTP client source seal differs")
    client_sha = http_sources[0].get("sha256")
    if type(client_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", client_sha) is None:
        fail("prepared HTTP client SHA-256 differs")
    client_raw = production.read_pinned_http_client_source(
        production.production_preflight_settings(),
        prepared.git_anchor,
        expected_sha256=client_sha,
    )
    direct = orchestrator._load_hyphenated_module(
        "_sq8_full_campaign_direct_identity",
        TOOLS_DIR / "run-sq8-direct-cancel-gate.py",
    )
    normal = direct.capture_service_identity()
    cached_epoch = prepared.config.normal_epoch
    cached_live = prepared.identity.live_identity
    if (
        normal.gateway_pid != cached_epoch.gateway_pid
        or normal.worker_pid != cached_epoch.worker_pid
        or normal.boot_id != prepared.config.boot_id
        or normal.n_restarts != prepared.identity.service_n_restarts
        or normal.unit != cached_live.service_unit
        or normal.user != cached_live.service_user
        or normal.uid != cached_live.service_uid
        or normal.gid != cached_live.service_gid
        or normal.control_group != cached_live.control_group
        or normal.gateway_pid != cached_live.gateway.pid
        or normal.gateway_starttime_ticks != cached_live.gateway.starttime_ticks
        or normal.worker_pid != cached_live.worker.pid
        or normal.worker_starttime_ticks != cached_live.worker.starttime_ticks
        or normal.n_restarts != cached_live.n_restarts
        or normal.boot_id != cached_live.boot_id
        or cached_live.derived_image_id != PRODUCTION_IMAGE_ID
        or cached_live.docker_network_id != PRODUCTION_NETWORK_ID
        or cached_live.docker_network_subnet != "172.20.0.0/16"
        or cached_live.docker_network_gateway != "172.20.0.1"
    ):
        fail("live normal service identity differs from the prepared campaign")

    bridge = SystemCampaignBridge(
        SystemBridgeInputs(
            prepared.identity,
            prepared.resource,
            prepared.git_anchor,
            prepared.secret_owner,
            client_raw,
            client_sha,
            production.production_preflight_settings().repo_root,
            collector.AMD_SMI_BIN,
        ),
        system_bridge_factories(collector, campaign_module, orchestrator),
        scan_evidence=prepared.secret_guard.reject,
    )
    binding_inputs = ProductionBindingInputs(
        environment,
        identity_module.SOURCE_ROLE_PATHS,
        production.production_preflight_settings().repo_root,
        {
            "api_contract": api_ingest.ApiContractInputBindings,
            "combined": openwebui_ingest.GateInputBindings,
            "direct_cancel": openwebui_ingest.DirectCancelInputBindings,
            "stop": stop_ingest.StopGateInputBindings,
            "failure": failure_ingest.FailureGateInputBindings,
            "latency": latency_ingest.LatencyGateInputBindings,
        },
        normal,
        direct.capture_service_identity,
        prepared.secret_guard,
        prepared.secret_guard.secrets,
        PRODUCTION_IMAGE_ID,
        PRODUCTION_NETWORK_ID,
        "172.20.0.0/16",
        "172.20.0.1",
        PRODUCTION_BROWSER_IMAGE_ID,
        PRODUCTION_BROWSER_IMAGE_ID,
        PRODUCTION_PROBE_IMAGE_ID,
        PRODUCTION_PROBE_IMAGE_ID,
        "http://192.168.0.66:3000/",
        "ullm-openai.service",
        "homelab1",
        prepared.config.boot_id,
        prepared.config.uid,
        prepared.config.gid,
    )
    bindings = production_binding_factory(binding_inputs)
    backend = ProductionCampaignBackend(
        bridge=bridge,
        runner=BoundedGateRunner(),
        secrets=SecretMasterPaths.from_owner(prepared.secret_owner),
        deployment=GateDeployment(
            PRODUCTION_IMAGE_ID,
            PRODUCTION_NETWORK_ID,
            PRODUCTION_BROWSER_IMAGE_ID,
            PRODUCTION_PROBE_IMAGE_ID,
            "http://192.168.0.66:3000/",
            "http://172.20.0.1:8000/readyz",
            "open-webui-network",
            "ullm-openai.service",
            Path("/run/user/1000/ullm-sq8-restart-epoch.pending"),
        ),
        bindings=bindings,
        ingestors=production_gate_ingestors(
            IngestorModules(
                api_ingest,
                openwebui_ingest,
                stop_ingest,
                failure_ingest,
                latency_ingest,
            )
        ),
    )
    validator = runtime.validator.FullCampaignIndependentValidator(
        expected_commit=prepared.request.expected_commit,
        expected_worker_binary_sha256=(prepared.request.expected_worker_binary_sha256),
        repo_root=production.production_preflight_settings().repo_root,
        forbidden_values=prepared.secret_guard.secrets,
    )
    published = orchestrator.run_full_campaign(
        prepared.config,
        backend,
        renderer_module.FullCampaignRenderer(),
        validator,
    )
    if not isinstance(published, os.PathLike):
        fail("production campaign publication path type differs")
    return Path(published)


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

EXPECTED_BINDING_FIELDS: Mapping[GateName, tuple[str, ...]] = {
    "api_contract": (
        "gate_source",
        "gate_source_sha256",
        "direct_source",
        "direct_source_sha256",
        "collector_source",
        "collector_source_sha256",
        "http_client_source",
        "http_client_source_sha256",
        "gateway_app_source",
        "gateway_app_source_sha256",
        "gateway_errors_source",
        "gateway_errors_source_sha256",
        "gateway_schemas_source",
        "gateway_schemas_source_sha256",
        "http_image_id",
        "docker_network_id",
        "service_unit",
        "service_user",
        "boot_id",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "restart_count",
        "uid",
        "gid",
        "forbidden_values",
    ),
    "combined": (
        "gate_source",
        "gate_source_sha256",
        "support_source",
        "support_source_sha256",
        "browser_script",
        "browser_script_sha256",
        "browser_image_reference",
        "browser_image_content_id",
        "openwebui_base_url",
        "service_unit",
        "boot_id",
        "gateway_pid",
        "uid",
        "gid",
        "restart_count",
        "forbidden_values",
    ),
    "direct_cancel": (
        "gate_source",
        "gate_source_sha256",
        "collector_source",
        "collector_source_sha256",
        "http_client_source",
        "http_client_source_sha256",
        "http_image_id",
        "docker_network_id",
        "service_unit",
        "service_user",
        "boot_id",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "restart_count",
        "uid",
        "gid",
        "forbidden_values",
    ),
    "stop": (
        "gate_source",
        "gate_source_sha256",
        "browser_script",
        "browser_script_sha256",
        "browser_image_reference",
        "browser_image_content_id",
        "openwebui_url",
        "service_unit",
        "service_user",
        "boot_id",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "restart_count",
        "uid",
        "gid",
        "forbidden_values",
    ),
    "failure": (
        "gate_source",
        "gate_source_sha256",
        "hook_source",
        "hook_source_sha256",
        "browser_source",
        "browser_source_sha256",
        "browser_image_reference",
        "browser_image_content_digest",
        "probe_image_reference",
        "probe_image_content_digest",
        "docker_network_id",
        "docker_network_subnet",
        "docker_network_gateway",
        "service_unit",
        "service_user",
        "boot_id",
        "control_group",
        "normal_gateway_pid",
        "normal_gateway_starttime_ticks",
        "normal_worker_pid",
        "normal_worker_starttime_ticks",
        "normal_restart_count",
        "restart_gateway_pid",
        "restart_gateway_starttime_ticks",
        "restart_worker_pid",
        "restart_worker_starttime_ticks",
        "restart_restart_count",
        "uid",
        "gid",
        "forbidden_values",
    ),
    "latency": (
        "gate_source",
        "gate_source_sha256",
        "direct_source",
        "direct_source_sha256",
        "collector_source",
        "collector_source_sha256",
        "http_client_source",
        "http_client_source_sha256",
        "restart_epoch_file",
        "restart_epoch_sha256",
        "http_image_id",
        "docker_network_id",
        "service_unit",
        "service_user",
        "boot_id",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "restart_count",
        "uid",
        "gid",
        "forbidden_values",
    ),
}


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
        if errors:
            raise errors[0]
        self.closed = True


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
        gate_bundle = work_dir / "gate-bundle"
        if (
            not work_dir.is_absolute()
            or gate_bundle.exists()
            or gate_bundle.is_symlink()
        ):
            self.poisoned = True
            fail("production gate bundle destination differs")
        try:
            self.secrets.revalidate()
            self.runner.run_gate(gate, gate_bundle, self.secrets, self.deployment)
            result = self.ingestors.for_gate(gate)(
                gate_bundle, self.bindings.build(gate, gate_bundle)
            )
            if gate == "failure":
                confirm = getattr(self.bindings, "confirm_failure", None)
                if not callable(confirm):
                    fail("production failure binding cannot confirm the restart epoch")
                epoch_file = confirm(result, gate_bundle)
                self.deployment = dataclasses.replace(
                    self.deployment, expected_epoch_file=epoch_file
                )
            elif gate == "latency":
                confirm_latency = getattr(self.bindings, "confirm_latency", None)
                if not callable(confirm_latency):
                    fail("production latency binding cannot close the restart epoch")
                confirm_latency()
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
        cleanup_owners: list[Any] = []
        if callable(getattr(self.bindings, "close", None)):
            cleanup_owners.append(self.bindings)
        cleanup_owners.extend((self.bridge, *reversed(self.owners)))
        for owner in cleanup_owners:
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self.closed = True


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
