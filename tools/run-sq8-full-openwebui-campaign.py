#!/usr/bin/env python3
"""Fail-closed orchestration boundary for one full SQ8 OpenWebUI campaign."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NoReturn, Protocol, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sq8_full_campaign_bundle import (  # noqa: E402
    AtomicCampaignDirectory,
    FileEvidence,
    PREVALIDATION_ROOT_FILES,
)
from sq8_openwebui_campaign import PidEpoch  # noqa: E402


PHASE_ORDER = (
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
)
DERIVED_ARTIFACTS = frozenset(
    {
        "sampling-results.json",
        "cancel-results.json",
        "prefill-latency-results.json",
        "api-contract-results.json",
        "openwebui-smoke.json",
        "soak-results.json",
        "release-matrix.json",
        "summary.md",
        "SHA256SUMS",
    }
)
MAX_PNG_BYTES = 128 << 20


class FullCampaignError(RuntimeError):
    """A campaign failed before its evidence directory could be published."""


def fail(message: str) -> NoReturn:
    raise FullCampaignError(message)


class SessionWriterProtocol(Protocol):
    """The subset of ``collect-sq8-openwebui-release.SessionWriter`` used here."""

    counts: collections.Counter[str]
    sequence: int
    writer: "AtomicJsonlWriterProtocol"

    def append(
        self, record_type: str, phase: str, case_id: str | None, **fields: Any
    ) -> None: ...


class AtomicJsonlWriterProtocol(Protocol):
    def write_value(self, value: dict[str, Any]) -> None: ...

    def commit(self) -> None: ...

    def abort_close(self) -> None: ...


class ClaimedGatewayEventProtocol(Protocol):
    phase: str
    case_id: str
    fields: dict[str, Any]


class CampaignJournalProtocol(Protocol):
    def start(self) -> str: ...

    def checkpoint(self, phase: str, deadline_ns: int) -> str: ...

    def arm_restart_transition(self) -> None: ...

    def claim_bundle_records(
        self, claims: Iterable[Any], deadline_ns: int
    ) -> list[ClaimedGatewayEventProtocol]: ...

    def confirm_restart_epoch(self, epoch: PidEpoch) -> None: ...

    def seal(self, expected_final_cursor: str, deadline_ns: int) -> str: ...

    def abort(self) -> None: ...


class ResourceSegmentResultProtocol(Protocol):
    segment: str
    identity: Any
    sampling_cases: tuple[dict[str, Any], ...]


class ResourceAdapterProtocol(Protocol):
    """A wrapper around the existing ``ResourceSegmentCollector`` instance."""

    def collect_normal(
        self, *, expected_identity: Any | None = None
    ) -> ResourceSegmentResultProtocol: ...

    def collect_restart(
        self, normal_identity: Any, *, expected_identity: Any | None = None
    ) -> ResourceSegmentResultProtocol: ...

    def close(self) -> None: ...


class ApiResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    derived_view: dict[str, Any]
    final_journal_cursor: str


class CombinedResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


class DirectResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


class ScreenshotProtocol(Protocol):
    path: Path
    bytes: int
    sha256: str


class StopResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    screenshot_evidence: ScreenshotProtocol
    derived_view: dict[str, Any]


class FailureScreenshotProtocol(Protocol):
    source_path: Path
    bundle_path: str
    bytes: int
    sha256: str


class FailureResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    fault_injection_record: dict[str, Any]
    lifecycle_claims: tuple[Any, ...]
    restart_probe_record: dict[str, Any]
    screenshot_evidence: FailureScreenshotProtocol
    derived_view: dict[str, Any]


class LatencyResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PreflightPhaseResult:
    environment_bytes: bytes
    model_identity_bytes: bytes
    header_fields: dict[str, Any]
    resource_header: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class FinalPhaseResult:
    lifecycle_probe_record: dict[str, Any]
    completed_utc: str
    completed_monotonic_ns: int
    final_git_commit: str
    final_git_status_raw: str


@dataclasses.dataclass(frozen=True)
class CampaignConfig:
    final_path: Path
    uid: int
    gid: int
    boot_id: str
    normal_epoch: PidEpoch
    operation_timeout_ns: int = 10_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.final_path, os.PathLike):
            fail("campaign final path type differs")
        if type(self.uid) is not int or self.uid < 0:
            fail("campaign UID binding differs")
        if type(self.gid) is not int or self.gid < 0:
            fail("campaign GID binding differs")
        if type(self.boot_id) is not str or len(self.boot_id) != 32:
            fail("campaign boot ID binding differs")
        try:
            int(self.boot_id, 16)
        except ValueError:
            fail("campaign boot ID syntax differs")
        if self.boot_id != self.boot_id.lower():
            fail("campaign boot ID must be lowercase")
        if not isinstance(self.normal_epoch, PidEpoch):
            fail("campaign normal PID epoch type differs")
        if type(self.operation_timeout_ns) is not int or self.operation_timeout_ns < 1:
            fail("campaign timeout binding differs")


@dataclasses.dataclass(frozen=True)
class CampaignEvidence:
    preflight: PreflightPhaseResult
    api_contract: ApiResultProtocol
    combined: CombinedResultProtocol
    direct_cancel: DirectResultProtocol
    stop: StopResultProtocol
    resource_normal: ResourceSegmentResultProtocol
    failure: FailureResultProtocol
    resource_restart: ResourceSegmentResultProtocol
    latency: LatencyResultProtocol
    final: FinalPhaseResult


@dataclasses.dataclass(frozen=True)
class RenderContext:
    stage_path: Path
    evidence: CampaignEvidence


class ViewsRenderer(Protocol):
    """Boundary implemented by ``sq8_full_campaign_views.py`` when wired."""

    def render(self, context: RenderContext) -> Mapping[str, bytes]: ...


class IndependentValidator(Protocol):
    """Validate the sealed prevalidation set and exclusively write validation."""

    def validate(self, stage_path: Path) -> FileEvidence: ...


class CampaignBackend(Protocol):
    """All command construction and live operations stay behind this boundary."""

    def now_ns(self) -> int: ...

    def scan_evidence(self, raw: bytes, label: str) -> None: ...

    def make_session_writer(self, path: Path) -> SessionWriterProtocol: ...

    def make_resource_writer(self, path: Path) -> AtomicJsonlWriterProtocol: ...

    def make_journal_capture(
        self, path: Path, boot_id: str, normal_epoch: PidEpoch
    ) -> CampaignJournalProtocol: ...

    def preflight(self, work_dir: Path) -> PreflightPhaseResult: ...

    def api_contract(self, work_dir: Path) -> ApiResultProtocol: ...

    def combined(self, work_dir: Path) -> CombinedResultProtocol: ...

    def direct_cancel(self, work_dir: Path) -> DirectResultProtocol: ...

    def stop(self, work_dir: Path) -> StopResultProtocol: ...

    def make_resource_adapter(
        self,
        *,
        normal_work_dir: Path,
        restart_work_dir: Path,
        stage_path: Path,
        session: SessionWriterProtocol,
        resource: AtomicJsonlWriterProtocol,
        journal: CampaignJournalProtocol,
    ) -> ResourceAdapterProtocol: ...

    def failure(self, work_dir: Path) -> FailureResultProtocol: ...

    def latency(self, work_dir: Path) -> LatencyResultProtocol: ...

    def final(self, work_dir: Path) -> FinalPhaseResult: ...

    def close(self) -> None: ...


BundleFactory = Callable[..., AtomicCampaignDirectory]


class _CampaignRuntimeGuard:
    """Close every live runtime owner before the bundle removes its stage."""

    def __init__(self, backend: CampaignBackend):
        self.backend = backend
        self.resource_adapter: ResourceAdapterProtocol | None = None
        self.journal: CampaignJournalProtocol | None = None
        self.session: SessionWriterProtocol | None = None
        self.resource: AtomicJsonlWriterProtocol | None = None
        self.backend_closed = False
        self.completed = False

    def __enter__(self) -> _CampaignRuntimeGuard:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        if self.completed:
            return
        cleanup_errors: list[BaseException] = []

        def attempt(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)

        if self.resource_adapter is not None:
            attempt(self.resource_adapter.close)
        if not self.backend_closed:
            attempt(self.backend.close)
        if self.resource is not None:
            attempt(self.resource.abort_close)
        if self.session is not None:
            attempt(self.session.writer.abort_close)
        if self.journal is not None:
            attempt(self.journal.abort)

        if not cleanup_errors:
            return
        if error is not None:
            error.add_note(
                "campaign cleanup also failed after every cleanup owner was attempted"
            )
            return
        raise FullCampaignError("campaign runtime cleanup failed") from cleanup_errors[
            0
        ]


def _deadline(backend: CampaignBackend, config: CampaignConfig) -> int:
    now = backend.now_ns()
    if type(now) is not int or now < 0:
        fail("campaign backend clock differs")
    return now + config.operation_timeout_ns


def _append_hook_record(
    session: SessionWriterProtocol,
    record: dict[str, Any],
    *,
    expected_phase: str,
    expected_type: str | None = None,
) -> None:
    if type(record) is not dict or set(record) != {
        "record_type",
        "phase",
        "case_id",
        "fields",
    }:
        fail("campaign hook record shape differs")
    record_type = record["record_type"]
    phase = record["phase"]
    case_id = record["case_id"]
    fields = record["fields"]
    if (
        type(record_type) is not str
        or (expected_type is not None and record_type != expected_type)
        or phase != expected_phase
        or type(case_id) is not str
        or not case_id
        or type(fields) is not dict
    ):
        fail("campaign hook record identity differs")
    session.append(record_type, phase, case_id, **fields)


def _append_claimed(
    session: SessionWriterProtocol,
    claimed: Iterable[ClaimedGatewayEventProtocol],
    *,
    expected_phase: str,
) -> None:
    for item in claimed:
        if (
            item.phase != expected_phase
            or type(item.case_id) is not str
            or not item.case_id
            or type(item.fields) is not dict
        ):
            fail("claimed campaign lifecycle identity differs")
        session.append("gateway_event", item.phase, item.case_id, **item.fields)


def _claim_phase(
    journal: CampaignJournalProtocol,
    session: SessionWriterProtocol,
    claims: Iterable[Any],
    *,
    expected_phase: str,
    deadline_ns: int,
) -> tuple[ClaimedGatewayEventProtocol, ...]:
    materialized = tuple(claims)
    claimed = tuple(journal.claim_bundle_records(materialized, deadline_ns))
    if len(claimed) != len(materialized):
        fail("campaign journal claim cardinality differs")
    _append_claimed(session, claimed, expected_phase=expected_phase)
    return claimed


def _checkpoint(
    journal: CampaignJournalProtocol,
    backend: CampaignBackend,
    config: CampaignConfig,
    phase: str,
) -> str:
    return journal.checkpoint(phase, _deadline(backend, config))


_IDENTITY_FIELDS = (
    "control_group",
    "gateway_pid",
    "gateway_starttime_ticks",
    "worker_pid",
    "worker_starttime_ticks",
    "n_restarts",
)


def _identity_values(identity: Any) -> tuple[Any, ...]:
    try:
        values = tuple(getattr(identity, field) for field in _IDENTITY_FIELDS)
    except AttributeError:
        fail("resource process identity shape differs")
    if (
        type(values[0]) is not str
        or not values[0]
        or any(type(value) is not int or value < 1 for value in values[1:5])
        or type(values[5]) is not int
        or values[5] < 0
    ):
        fail("resource process identity values differ")
    return values


def _restart_identity_from_probe(normal_identity: Any, fields: dict[str, Any]) -> Any:
    if set(_IDENTITY_FIELDS) - set(fields):
        fail("restart probe lacks a process identity")
    values = {field: fields[field] for field in _IDENTITY_FIELDS}
    try:
        restart = type(normal_identity)(**values)
    except (TypeError, ValueError):
        fail("restart probe cannot bind the resource identity type")
    normal_values = _identity_values(normal_identity)
    restart_values = _identity_values(restart)
    if (
        restart_values[0] != normal_values[0]
        or restart_values[1] == normal_values[1]
        or restart_values[3] == normal_values[3]
        or restart_values[5] != normal_values[5] + 1
    ):
        fail("restart probe process epoch differs")
    return restart


def _validate_resource_result(
    session: SessionWriterProtocol,
    result: ResourceSegmentResultProtocol,
    *,
    segment: str,
    prior_probe_count: int,
) -> None:
    if result.segment != segment:
        fail("resource adapter segment result differs")
    _identity_values(result.identity)
    try:
        sampling_cases = result.sampling_cases
    except AttributeError:
        fail("resource adapter result lacks sampling cases")
    expected_sampling_count = 20 if segment == "normal" else 0
    if (
        type(sampling_cases) is not tuple
        or len(sampling_cases) != expected_sampling_count
        or any(type(item) is not dict for item in sampling_cases)
    ):
        fail("resource adapter sampling result differs")
    if session.counts.get("lifecycle_probe", 0) != prior_probe_count + 1:
        fail("resource adapter did not append exactly one lifecycle probe")


def _validate_ready_probe_record(
    record: dict[str, Any],
    *,
    phase: str,
    name: str,
    expected_identity: Any | None = None,
) -> dict[str, Any]:
    if (
        type(record) is not dict
        or set(record) != {"record_type", "phase", "case_id", "fields"}
        or record["record_type"] != "lifecycle_probe"
        or record["phase"] != phase
        or record["case_id"] != name
        or type(record["fields"]) is not dict
    ):
        fail("campaign readiness probe record differs")
    fields = record["fields"]
    if set(fields) != {
        "probe",
        "observed_monotonic_ns",
        "service_active",
        "ready_http_status",
        *_IDENTITY_FIELDS,
    }:
        fail("campaign readiness probe fields differ")
    if (
        fields["probe"] != name
        or type(fields["observed_monotonic_ns"]) is not int
        or fields["observed_monotonic_ns"] < 0
        or fields["service_active"] is not True
        or fields["ready_http_status"] != 200
    ):
        fail("campaign readiness probe state differs")
    if expected_identity is not None and tuple(
        fields[field] for field in _IDENTITY_FIELDS
    ) != _identity_values(expected_identity):
        fail("campaign readiness probe process identity differs")
    return fields


def _copy_stop_screenshot(
    bundle: AtomicCampaignDirectory,
    result: StopResultProtocol,
    backend: CampaignBackend,
) -> None:
    evidence = result.screenshot_evidence
    bundle.copy_file(
        evidence.path,
        "browser/openwebui-stop-before.png",
        expected_bytes=evidence.bytes,
        expected_sha256=evidence.sha256,
        maximum_bytes=MAX_PNG_BYTES,
        scan=backend.scan_evidence,
    )


def _copy_failure_screenshot(
    bundle: AtomicCampaignDirectory,
    result: FailureResultProtocol,
    backend: CampaignBackend,
) -> None:
    evidence = result.screenshot_evidence
    if evidence.bundle_path != "browser/post-header-failure.png":
        fail("failure screenshot bundle path differs")
    bundle.copy_file(
        evidence.source_path,
        evidence.bundle_path,
        expected_bytes=evidence.bytes,
        expected_sha256=evidence.sha256,
        maximum_bytes=MAX_PNG_BYTES,
        scan=backend.scan_evidence,
    )


def _failure_phase(
    *,
    result: FailureResultProtocol,
    journal: CampaignJournalProtocol,
    session: SessionWriterProtocol,
    normal_identity: Any,
    backend: CampaignBackend,
    config: CampaignConfig,
) -> Any:
    actions = tuple(result.browser_action_records)
    claims = tuple(result.lifecycle_claims)
    if len(actions) != 9 or len(claims) != 10:
        fail("post-header failure action or lifecycle cardinality differs")
    if [getattr(item, "case_id", None) for item in claims] != (
        ["post-header-failure"] * 5 + ["post-header-recovery"] * 5
    ):
        fail("post-header failure lifecycle partition differs")

    for record in actions[:4]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    claimed = tuple(journal.claim_bundle_records(claims, _deadline(backend, config)))
    if len(claimed) != 10:
        fail("post-header failure global journal claim cardinality differs")
    _append_claimed(session, claimed[:4], expected_phase="post_header_failure")
    _append_hook_record(
        session,
        result.fault_injection_record,
        expected_phase="post_header_failure",
        expected_type="fault_injection",
    )
    _append_claimed(session, claimed[4:5], expected_phase="post_header_failure")
    _append_hook_record(
        session,
        actions[4],
        expected_phase="post_header_failure",
        expected_type="browser_action",
    )
    probe = result.restart_probe_record
    probe_fields = _validate_ready_probe_record(
        probe,
        phase="post_header_failure",
        name="post-header-restart-ready",
    )
    restart_identity = _restart_identity_from_probe(normal_identity, probe_fields)
    restart_epoch = PidEpoch(
        restart_identity.gateway_pid,
        restart_identity.worker_pid,
    )
    journal.confirm_restart_epoch(restart_epoch)
    _append_hook_record(
        session,
        probe,
        expected_phase="post_header_failure",
        expected_type="lifecycle_probe",
    )
    for record in actions[5:7]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    _append_claimed(session, claimed[5:], expected_phase="post_header_failure")
    for record in actions[7:]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    return restart_identity


def _render_artifacts(
    bundle: AtomicCampaignDirectory,
    renderer: ViewsRenderer,
    context: RenderContext,
    backend: CampaignBackend,
) -> None:
    rendered = renderer.render(context)
    if type(rendered) is not dict or set(rendered) != set(DERIVED_ARTIFACTS):
        fail("full campaign rendered artifact set differs")
    for relative in sorted(rendered, key=lambda item: item.encode("utf-8")):
        raw = rendered[relative]
        if type(raw) is not bytes or not raw:
            fail("full campaign rendered artifact bytes differ")
        bundle.write_bytes(relative, raw, scan=backend.scan_evidence)


def _seal_private_runtime_artifact(path: Path, *, uid: int, gid: int) -> None:
    """Tighten files produced by the existing collectors to the bundle's 0600 mode."""

    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or before.st_size < 1
        ):
            fail("campaign runtime artifact identity differs")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        entry = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (entry.st_dev, entry.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_nlink != 1
            or after.st_uid != uid
            or after.st_gid != gid
        ):
            fail("campaign runtime artifact changed while sealing")
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except FullCampaignError:
        raise
    except OSError:
        fail("failed to seal a campaign runtime artifact")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail("failed to close a campaign runtime artifact")


def run_full_campaign(
    config: CampaignConfig,
    backend: CampaignBackend,
    renderer: ViewsRenderer,
    validator: IndependentValidator,
    *,
    bundle_factory: BundleFactory = AtomicCampaignDirectory,
) -> Path:
    """Run one serial campaign and publish only independently validated evidence."""

    cleanup = _CampaignRuntimeGuard(backend)
    try:
        with (
            bundle_factory(
                config.final_path,
                uid=config.uid,
                gid=config.gid,
            ) as bundle,
            cleanup,
        ):
            session = backend.make_session_writer(
                bundle.artifact_path("raw-session-results.jsonl")
            )
            cleanup.session = session
            resource = backend.make_resource_writer(
                bundle.artifact_path("soak-resources.raw.jsonl")
            )
            cleanup.resource = resource
            journal = backend.make_journal_capture(
                bundle.artifact_path("service-journal.raw.jsonl"),
                config.boot_id,
                config.normal_epoch,
            )
            cleanup.journal = journal
            journal.start()

            preflight = backend.preflight(bundle.component_directory("preflight"))
            if not isinstance(preflight, PreflightPhaseResult):
                fail("preflight result type differs")
            bundle.write_bytes(
                "environment.json",
                preflight.environment_bytes,
                scan=backend.scan_evidence,
            )
            bundle.write_bytes(
                "model-identity.json",
                preflight.model_identity_bytes,
                scan=backend.scan_evidence,
            )
            session.append("header", "preflight", None, **preflight.header_fields)
            resource.write_value(preflight.resource_header)
            _checkpoint(journal, backend, config, "preflight")

            api = backend.api_contract(bundle.component_directory("api-contract"))
            for record in api.http_records:
                _append_hook_record(session, record, expected_phase="api_contract")
            api_cursor = _checkpoint(journal, backend, config, "api_contract")
            if api.final_journal_cursor != api_cursor:
                fail("API contract journal boundary differs from the global campaign")

            combined = backend.combined(bundle.component_directory("combined"))
            for record in combined.browser_action_records:
                _append_hook_record(session, record, expected_phase="openwebui")
            _claim_phase(
                journal,
                session,
                combined.lifecycle_claims,
                expected_phase="openwebui",
                deadline_ns=_deadline(backend, config),
            )
            _checkpoint(journal, backend, config, "openwebui")

            direct = backend.direct_cancel(bundle.component_directory("direct-cancel"))
            for record in direct.http_records:
                _append_hook_record(session, record, expected_phase="cancellation")
            _claim_phase(
                journal,
                session,
                direct.lifecycle_claims,
                expected_phase="cancellation",
                deadline_ns=_deadline(backend, config),
            )
            stop = backend.stop(bundle.component_directory("stop"))
            for record in stop.browser_action_records:
                _append_hook_record(session, record, expected_phase="cancellation")
            _claim_phase(
                journal,
                session,
                stop.lifecycle_claims,
                expected_phase="cancellation",
                deadline_ns=_deadline(backend, config),
            )
            _copy_stop_screenshot(bundle, stop, backend)
            _checkpoint(journal, backend, config, "cancellation")

            normal_work = bundle.component_directory("resource-normal")
            failure_work = bundle.component_directory("failure")
            restart_work = bundle.component_directory("resource-restart")
            resource_adapter = backend.make_resource_adapter(
                normal_work_dir=normal_work,
                restart_work_dir=restart_work,
                stage_path=bundle.stage_path,
                session=session,
                resource=resource,
                journal=journal,
            )
            cleanup.resource_adapter = resource_adapter
            probe_count = session.counts.get("lifecycle_probe", 0)
            normal_resource = resource_adapter.collect_normal()
            _validate_resource_result(
                session,
                normal_resource,
                segment="normal",
                prior_probe_count=probe_count,
            )
            normal_values = _identity_values(normal_resource.identity)
            if normal_values[1] != config.normal_epoch.gateway_pid or normal_values[
                3
            ] != (config.normal_epoch.worker_pid):
                fail("normal resource identity differs from the journal epoch")
            _checkpoint(journal, backend, config, "resource_normal")
            journal.arm_restart_transition()

            failure = backend.failure(failure_work)
            _copy_failure_screenshot(bundle, failure, backend)
            restart_identity = _failure_phase(
                result=failure,
                journal=journal,
                session=session,
                normal_identity=normal_resource.identity,
                backend=backend,
                config=config,
            )
            _checkpoint(journal, backend, config, "post_header_failure")

            probe_count = session.counts.get("lifecycle_probe", 0)
            restart_resource = resource_adapter.collect_restart(
                normal_resource.identity,
                expected_identity=restart_identity,
            )
            _validate_resource_result(
                session,
                restart_resource,
                segment="restart",
                prior_probe_count=probe_count,
            )
            if _identity_values(restart_resource.identity) != _identity_values(
                restart_identity
            ):
                fail("restart resource identity differs from the confirmed epoch")
            resource_adapter.close()
            cleanup.resource_adapter = None
            _checkpoint(journal, backend, config, "resource_restart")

            latency = backend.latency(bundle.component_directory("latency"))
            for record in latency.http_records:
                _append_hook_record(session, record, expected_phase="latency")
            _claim_phase(
                journal,
                session,
                latency.lifecycle_claims,
                expected_phase="latency",
                deadline_ns=_deadline(backend, config),
            )
            _checkpoint(journal, backend, config, "latency")

            final = backend.final(bundle.component_directory("final"))
            backend.close()
            cleanup.backend_closed = True
            final_probe_fields = _validate_ready_probe_record(
                final.lifecycle_probe_record,
                phase="final",
                name="final-service-ready",
                expected_identity=restart_identity,
            )
            if (
                type(final.completed_utc) is not str
                or not final.completed_utc
                or type(final.completed_monotonic_ns) is not int
                or final.completed_monotonic_ns
                < final_probe_fields["observed_monotonic_ns"]
                or type(final.final_git_commit) is not str
                or len(final.final_git_commit) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in final.final_git_commit
                )
            ):
                fail("final campaign metadata differs")
            _append_hook_record(
                session,
                final.lifecycle_probe_record,
                expected_phase="final",
                expected_type="lifecycle_probe",
            )
            if session.counts.get("lifecycle_probe", 0) != 4:
                fail("full campaign must contain exactly four lifecycle probes")
            final_cursor = _checkpoint(journal, backend, config, "final")
            counts = dict(session.counts)
            counts["run_end"] = counts.get("run_end", 0) + 1
            status_raw = final.final_git_status_raw
            if type(status_raw) is not str:
                fail("final Git status type differs")
            session.append(
                "run_end",
                "final",
                None,
                completed_utc=final.completed_utc,
                completed_monotonic_ns=final.completed_monotonic_ns,
                final_git_commit=final.final_git_commit,
                final_git_status_raw=status_raw,
                final_git_status_sha256=hashlib.sha256(
                    status_raw.encode("utf-8", errors="strict")
                ).hexdigest(),
                record_counts=counts,
                final_journal_cursor=final_cursor,
            )
            resource.commit()
            session.writer.commit()
            journal.seal(final_cursor, _deadline(backend, config))
            for relative in (
                "raw-session-results.jsonl",
                "soak-resources.raw.jsonl",
                "service-journal.raw.jsonl",
                "amd-smi-metric-normal-before.json",
                "amd-smi-metric-normal-after.json",
                "amd-smi-metric-restart-before.json",
                "amd-smi-metric-restart-after.json",
            ):
                _seal_private_runtime_artifact(
                    bundle.artifact_path(relative), uid=config.uid, gid=config.gid
                )

            evidence = CampaignEvidence(
                preflight,
                api,
                combined,
                direct,
                stop,
                normal_resource,
                failure,
                restart_resource,
                latency,
                final,
            )
            _render_artifacts(
                bundle,
                renderer,
                RenderContext(bundle.stage_path, evidence),
                backend,
            )
            if set(PREVALIDATION_ROOT_FILES) - {
                path.name for path in bundle.stage_path.iterdir() if path.is_file()
            }:
                fail("full campaign prevalidation artifacts are incomplete")
            bundle.clear_component_work()
            bundle.validate_before_independent_validator()
            validation_evidence = validator.validate(bundle.stage_path)
            published = bundle.publish(validation_evidence)
            if not isinstance(published, os.PathLike):
                fail("campaign publication path type differs")
            cleanup.completed = True
            return Path(published)
    except BaseException:
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-backend",
        action="store_true",
        help="reserved until the production backend is explicitly wired",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(
        "full campaign production backend is not wired; refusing to run",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignBackend",
    "CampaignConfig",
    "CampaignEvidence",
    "FinalPhaseResult",
    "FileEvidence",
    "FullCampaignError",
    "IndependentValidator",
    "PHASE_ORDER",
    "PreflightPhaseResult",
    "RenderContext",
    "ResourceAdapterProtocol",
    "ViewsRenderer",
    "run_full_campaign",
]
