#!/usr/bin/env python3
"""Production backend composition for the serial SQ8 OpenWebUI campaign."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import selectors
import signal
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
            identities = inputs.resource.session_header_fields["identities"]
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


class ProductionGateBindingsFactory:
    """Build exact ingestor bindings and pin the one post-failure epoch."""

    def __init__(
        self,
        normal_identity: Any,
        builders: Mapping[GateName, Callable[[GateBindingContext], Any]],
        restart_extractor: Callable[[Any, Path], tuple[Any, Path, str]],
    ) -> None:
        if set(builders) != set(GATE_ORDER):
            fail("production gate binding builder set differs")
        self.normal_identity = normal_identity
        self.builders = dict(builders)
        self.restart_extractor = restart_extractor
        self.restart_identity: Any | None = None
        self.restart_epoch_file: Path | None = None
        self.restart_epoch_sha256: str | None = None

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
        identity, epoch_file, epoch_sha256 = self.restart_extractor(result, work_dir)
        self.confirm_restart(identity, epoch_file, epoch_sha256)
        return epoch_file

    def build(self, gate: GateName, work_dir: Path) -> Any:
        if gate == "latency" and self.restart_identity is None:
            fail("latency binding lacks the confirmed restart epoch")
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
            if gate == "failure":
                confirm = getattr(self.bindings, "confirm_failure", None)
                if not callable(confirm):
                    fail("production failure binding cannot confirm the restart epoch")
                epoch_file = confirm(result, work_dir)
                self.deployment = dataclasses.replace(
                    self.deployment, expected_epoch_file=epoch_file
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
