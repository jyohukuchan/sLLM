from __future__ import annotations

import collections
import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORCHESTRATOR = load_module(
    "test_run_sq8_full_openwebui_campaign_tool",
    TOOLS / "run-sq8-full-openwebui-campaign.py",
)
CAMPAIGN = load_module(
    "test_run_sq8_full_openwebui_campaign_journal",
    TOOLS / "sq8_openwebui_campaign.py",
)
COLLECTOR = load_module(
    "test_run_sq8_full_openwebui_campaign_collector",
    TOOLS / "collect-sq8-openwebui-release.py",
)


BOOT_ID = "0123456789abcdef0123456789abcdef"
NORMAL_GATEWAY = 1001
NORMAL_WORKER = 1002
RESTART_GATEWAY = 2001
RESTART_WORKER = 2002


class BlockingJournalSource:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._rows: collections.deque[bytes] = collections.deque()
        self.closed = False

    def open_after(self, unit: str, boot_id: str) -> str:
        if unit != "ullm-openai.service" or boot_id != BOOT_ID:
            raise AssertionError("journal source binding differs")
        return "anchor-cursor"

    def read_next(self, timeout_usec: int) -> bytes | None:
        with self._condition:
            if not self._rows and not self.closed:
                self._condition.wait(timeout_usec / 1_000_000)
            if self._rows:
                return self._rows.popleft()
            return None

    def feed(self, *rows: bytes) -> None:
        with self._condition:
            self._rows.extend(rows)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()


class RecordingJournal:
    def __init__(self, capture, calls: list[str]):
        self.capture = capture
        self.calls = calls

    def start(self):
        self.calls.append("journal:start")
        return self.capture.start()

    def checkpoint(self, phase, deadline_ns):
        self.calls.append(f"checkpoint:{phase}")
        return self.capture.checkpoint(phase, deadline_ns)

    def arm_restart_transition(self):
        self.calls.append("journal:arm_restart")
        return self.capture.arm_restart_transition()

    def claim_bundle_records(self, claims, deadline_ns):
        materialized = tuple(claims)
        self.calls.append(
            "journal:claim:" + materialized[0].phase + f":{len(materialized)}"
        )
        return self.capture.claim_bundle_records(materialized, deadline_ns)

    def confirm_restart_epoch(self, epoch):
        self.calls.append(
            f"journal:confirm_restart:{epoch.gateway_pid}:{epoch.worker_pid}"
        )
        return self.capture.confirm_restart_epoch(epoch)

    def seal(self, cursor, deadline_ns):
        self.calls.append("journal:seal")
        return self.capture.seal(cursor, deadline_ns)

    def abort(self):
        self.calls.append("journal:abort")
        return self.capture.abort()


@dataclasses.dataclass(frozen=True)
class Identity:
    control_group: str
    gateway_pid: int
    gateway_starttime_ticks: int
    worker_pid: int
    worker_starttime_ticks: int
    n_restarts: int


def sampling_cases():
    return tuple(
        {
            "request_index": index,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": index,
            "http_status": 200,
            "http_outcome": "eof",
            "release_outcome": "length",
            "completion_tokens": 2,
            "reset_complete": True,
        }
        for index in range(5, 101, 5)
    )


def hook(record_type: str, phase: str, case_id: str, **fields):
    return {
        "record_type": record_type,
        "phase": phase,
        "case_id": case_id,
        "fields": fields,
    }


class FakeResourceAdapter:
    def __init__(self, backend, stage, session, resource, journal):
        self.backend = backend
        self.stage = stage
        self.session = session
        self.resource = resource
        self.journal = journal
        self.closed = False

    def _collect(self, segment: str, identity: Identity):
        self.backend._maybe_fail(f"resource_{segment}")
        self.backend.calls.append(f"resource:{segment}")
        phase = f"resource_{segment}"
        probe = f"{segment}-segment-start"
        self.session.append(
            "lifecycle_probe",
            phase,
            probe,
            **self.backend.probe_fields(probe, identity),
        )
        claims = self.backend.trace(phase, f"{segment}-request", identity.gateway_pid)
        self.backend.source.feed(*(claim.raw for claim in claims))
        claimed = self.journal.claim_bundle_records(
            claims, time.monotonic_ns() + 3_000_000_000
        )
        for item in claimed:
            self.session.append(
                "gateway_event", item.phase, item.case_id, **item.fields
            )
        self.resource.write_value(
            {
                "schema_version": "ullm.sq8.release_measurement.raw.v1",
                "record_type": "resource_sample",
                "segment": segment,
            }
        )
        for boundary in ("before", "after"):
            path = self.stage / f"amd-smi-metric-{segment}-{boundary}.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, b"{}\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        normal = segment == "normal"
        return COLLECTOR.ResourceSegmentResult(
            segment=segment,
            identity=identity,
            warmup_requests=10,
            measured_requests=100 if normal else 20,
            negative_requests=3 if normal else 0,
            resource_samples=505 if normal else 105,
            gpu_metrics=2,
            sampling_cases=sampling_cases() if normal else (),
        )

    def collect_normal(self, *, expected_identity=None):
        assert expected_identity is None
        return self._collect("normal", self.backend.normal_identity)

    def collect_restart(self, normal_identity, *, expected_identity=None):
        assert normal_identity == self.backend.normal_identity
        assert expected_identity == self.backend.restart_identity
        return self._collect("restart", self.backend.restart_identity)

    def close(self):
        if not self.closed:
            self.backend.calls.append("resource:close")
            self.closed = True
            if self.backend.adapter_close_fail:
                raise RuntimeError("injected resource close failure")


class FakeBackend:
    def __init__(
        self,
        root: Path,
        *,
        fail_phase: str | None = None,
        adapter_close_fail: bool = False,
    ):
        self.root = root
        self.fail_phase = fail_phase
        self.adapter_close_fail = adapter_close_fail
        self.calls: list[str] = []
        self.observer_open = True
        self.closed = False
        self.source = BlockingJournalSource()
        self.journal = None
        self.guard = COLLECTOR.SecretGuard(b"not-a-real-secret")
        self.counter = 1
        self.normal_identity = Identity(
            "/system.slice/ullm-openai.service", 1001, 10001, 1002, 10002, 2
        )
        self.restart_identity = Identity(
            "/system.slice/ullm-openai.service", 2001, 20001, 2002, 20002, 3
        )
        self.stop_png = root / "stop.png"
        self.failure_png = root / "failure.png"
        self.stop_png.write_bytes(b"\x89PNG\r\n\x1a\nstop")
        self.failure_png.write_bytes(b"\x89PNG\r\n\x1a\nfailure")

    def _maybe_fail(self, phase: str) -> None:
        if self.fail_phase == phase:
            raise RuntimeError(f"injected {phase} failure")

    def now_ns(self):
        return time.monotonic_ns()

    def scan_evidence(self, raw, label):
        self.guard.reject(raw, label)

    def make_session_writer(self, path):
        return COLLECTOR.SessionWriter(path, self.guard)

    def make_resource_writer(self, path):
        return COLLECTOR.AtomicJsonlWriter(path, self.guard)

    def make_journal_capture(self, path, boot_id, normal_epoch):
        capture = CAMPAIGN.CampaignJournalCapture(
            path,
            boot_id,
            CAMPAIGN.PidEpoch(normal_epoch.gateway_pid, normal_epoch.worker_pid),
            scan_raw=self.scan_evidence,
            source=self.source,
        )
        self.journal = RecordingJournal(capture, self.calls)
        return self.journal

    def preflight(self, work_dir):
        self._maybe_fail("preflight")
        self.calls.append("phase:preflight")
        return ORCHESTRATOR.PreflightPhaseResult(
            b'{"environment":true}\n',
            b'{"model":true}\n',
            {"run_id": "test-run", "marker": "preflight"},
            {
                "schema_version": "ullm.sq8.release_measurement.raw.v1",
                "record_type": "header",
            },
        )

    def api_contract(self, work_dir):
        self._maybe_fail("api_contract")
        self.calls.append("phase:api_contract")
        return types.SimpleNamespace(
            http_records=(
                hook(
                    "http_response_end",
                    "api_contract",
                    "api-case",
                    marker="api",
                ),
            ),
            journal_records=(
                hook(
                    "api_journal_observation",
                    "api_contract",
                    "api-journal-01",
                    observation_index=0,
                    journal_cursor="anchor-cursor",
                    journal_monotonic_usec=1,
                    journal_pid=NORMAL_GATEWAY,
                    message_utf8_bytes=3,
                    message_sha256=hashlib.sha256(b"api").hexdigest(),
                ),
            ),
            quiet_check_records=(
                hook(
                    "lifecycle_quiet_check",
                    "api_contract",
                    "api-quiet",
                    quiet_sequence=0,
                    label="api-quiet",
                    checked_monotonic_ns=1,
                    observer_open=False,
                    observer_event_count=0,
                    new_journal_record_count=1,
                    journal_record_count=1,
                    journal_cursor="anchor-cursor",
                ),
            ),
            derived_view={"api": True},
            final_journal_cursor="anchor-cursor",
        )

    def combined(self, work_dir):
        self._maybe_fail("openwebui")
        self.calls.append("phase:openwebui")
        claims = self.trace("openwebui", "openwebui-case", NORMAL_GATEWAY)
        self.source.feed(*(claim.raw for claim in claims))
        return types.SimpleNamespace(
            browser_action_records=(self.action("openwebui", "openwebui-case", 0),),
            lifecycle_claims=claims,
            derived_view={"openwebui": True},
        )

    def direct_cancel(self, work_dir):
        self._maybe_fail("direct_cancel")
        self.calls.append("phase:direct_cancel")
        claims = self.trace("cancellation", "direct-case", NORMAL_GATEWAY)
        self.source.feed(*(claim.raw for claim in claims))
        return types.SimpleNamespace(
            http_records=(
                hook(
                    "http_response_end", "cancellation", "direct-case", marker="direct"
                ),
            ),
            lifecycle_claims=claims,
            derived_view={"direct": True},
        )

    def stop(self, work_dir):
        self._maybe_fail("stop")
        self.calls.append("phase:stop")
        claims = self.trace("cancellation", "stop-case", NORMAL_GATEWAY)
        self.source.feed(*(claim.raw for claim in claims))
        raw = self.stop_png.read_bytes()
        return types.SimpleNamespace(
            browser_action_records=(self.action("cancellation", "stop-case", 0),),
            lifecycle_claims=claims,
            screenshot_evidence=types.SimpleNamespace(
                path=self.stop_png,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
            derived_view={"stop": True},
        )

    def make_resource_adapter(
        self,
        *,
        normal_work_dir,
        restart_work_dir,
        stage_path,
        session,
        resource,
        journal,
    ):
        self.calls.append("resource:open")
        return FakeResourceAdapter(self, stage_path, session, resource, journal)

    def failure(self, work_dir):
        self._maybe_fail("post_header_failure")
        self.calls.append("phase:post_header_failure")
        old = self.trace(
            "post_header_failure",
            "post-header-failure",
            NORMAL_GATEWAY,
            terminal="worker_fatal",
        )
        new = self.trace("post_header_failure", "post-header-recovery", RESTART_GATEWAY)
        claims = old + new
        self.source.feed(*(claim.raw for claim in claims))
        actions = tuple(
            self.action(
                "post_header_failure",
                "post-header-failure" if index < 5 else "post-header-recovery",
                index,
            )
            for index in range(9)
        )
        raw = self.failure_png.read_bytes()
        return types.SimpleNamespace(
            browser_action_records=actions,
            fault_injection_record=hook(
                "fault_injection",
                "post_header_failure",
                "post-header-failure",
                injection="post_header_worker_kill",
                target_pid=NORMAL_WORKER,
                target_starttime_ticks=10002,
                signal="SIGKILL",
                command="kill",
                started_monotonic_ns=1,
                completed_monotonic_ns=2,
            ),
            lifecycle_claims=claims,
            restart_probe_record=hook(
                "lifecycle_probe",
                "post_header_failure",
                "post-header-restart-ready",
                **self.probe_fields("post-header-restart-ready", self.restart_identity),
            ),
            screenshot_evidence=types.SimpleNamespace(
                source_path=self.failure_png,
                bundle_path="browser/post-header-failure.png",
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
            derived_view={"failure": True},
        )

    def latency(self, work_dir):
        self._maybe_fail("latency")
        self.calls.append("phase:latency")
        claims = self.trace("latency", "latency-case", RESTART_GATEWAY)
        self.source.feed(*(claim.raw for claim in claims))
        return types.SimpleNamespace(
            http_records=(
                hook("http_response_end", "latency", "latency-case", marker="latency"),
            ),
            lifecycle_claims=claims,
            derived_view={"latency": True},
        )

    def final(self, work_dir):
        self._maybe_fail("final")
        self.calls.append("phase:final")
        return ORCHESTRATOR.FinalPhaseResult(
            hook(
                "lifecycle_probe",
                "final",
                "final-service-ready",
                **self.probe_fields("final-service-ready", self.restart_identity),
            ),
            "2026-07-11T00:00:00Z",
            self.now_ns(),
            "a" * 40,
            "",
        )

    def close(self):
        if not self.closed:
            self.calls.append("backend:close")
            self.observer_open = False
            self.closed = True

    def action(self, phase: str, case_id: str, index: int):
        return hook(
            "browser_action",
            phase,
            case_id,
            browser_case=case_id,
            action_index=index,
            action="wait_visible",
            marker=f"action-{index}",
        )

    def probe_fields(self, name: str, identity: Identity):
        return {
            "probe": name,
            "observed_monotonic_ns": self.now_ns(),
            "service_active": True,
            "ready_http_status": 200,
            **dataclasses.asdict(identity),
        }

    def trace(
        self,
        phase: str,
        case_id: str,
        gateway_pid: int,
        *,
        terminal: str = "request_released",
    ):
        request_id = f"request-{self.counter}"
        completion_id = f"completion-{self.counter}"
        base = {
            "schema_version": "ullm.gateway.lifecycle.v1",
            "request_id": request_id,
            "completion_id": completion_id,
        }
        events = [
            {
                **base,
                "event": "request_admitted",
                "stream": True,
                "prompt_tokens": 1,
                "max_completion_tokens": 2,
            },
            {
                **base,
                "event": "request_started",
                "stream": True,
                "prompt_tokens": 1,
                "admit_to_start_ns": 10,
            },
            {
                **base,
                "event": "request_progress",
                "phase": "decode",
                "processed_prompt_tokens": 1,
                "prompt_tokens": 1,
            },
            {
                **base,
                "event": "request_first_token",
                "stream": True,
                "completion_tokens": 1,
            },
        ]
        if terminal == "worker_fatal":
            events.append(
                {
                    **base,
                    "event": "worker_fatal",
                    "reason": "worker_exit",
                    "admit_to_fatal_ns": 50,
                }
            )
        else:
            events.append(
                {
                    **base,
                    "event": "request_released",
                    "stream": True,
                    "outcome": "length",
                    "cancel_reason": None,
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reset_complete": True,
                    "admit_to_start_ns": 10,
                    "start_to_release_ns": 40,
                    "admit_to_release_ns": 50,
                }
            )
        claims = []
        for event in events:
            event["observed_monotonic_ns"] = self.counter * 1_000_000
            message = json.dumps(event, sort_keys=True, separators=(",", ":"))
            row = json.dumps(
                {
                    "__CURSOR": f"cursor-{self.counter}",
                    "__MONOTONIC_TIMESTAMP": str(self.counter * 1000),
                    "_BOOT_ID": BOOT_ID,
                    "_PID": str(gateway_pid),
                    "_SYSTEMD_UNIT": "ullm-openai.service",
                    "PRIORITY": "6",
                    "MESSAGE": message,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            claims.append(CAMPAIGN.BundleLifecycleClaim(row, phase, case_id))
            self.counter += 1
        return tuple(claims)


class FakeRenderer:
    def __init__(self, calls):
        self.calls = calls

    def render(self, context):
        self.calls.append("render")
        assert isinstance(
            context.evidence.resource_normal, COLLECTOR.ResourceSegmentResult
        )
        assert context.evidence.resource_normal.sampling_cases == sampling_cases()
        assert isinstance(
            context.evidence.resource_restart, COLLECTOR.ResourceSegmentResult
        )
        assert context.evidence.resource_restart.sampling_cases == ()
        return {
            relative: (b"summary\n" if relative.endswith(".md") else b"{}\n")
            for relative in ORCHESTRATOR.DERIVED_ARTIFACTS
        }


class FakeValidator:
    def __init__(self, calls, *, fail: bool = False):
        self.calls = calls
        self.fail = fail

    def validate(self, stage_path):
        self.calls.append("validate")
        work_roots = list(
            stage_path.parent.glob(f".{stage_path.name.split('.')[1]}.work-*")
        )
        if work_roots:
            assert all(not list(path.iterdir()) for path in work_roots)
        if self.fail:
            raise RuntimeError("injected validator failure")
        descriptor = os.open(
            stage_path / "release-validation.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            raw = b'{"validated":true}\n'
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return ORCHESTRATOR.FileEvidence(len(raw), hashlib.sha256(raw).hexdigest())


class ProductionSafetyPrimitiveTests(unittest.TestCase):
    API_SECRET = b"api-secret-0123456789abcdef"
    OPENWEBUI_TOKEN = b"openwebui-token-0123456789abcdef"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)

    def tearDown(self):
        self.temporary.cleanup()

    def write_secret(self, name: str, raw: bytes, mode: int) -> Path:
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(mode)
        return path

    def snapshot_api(self, path: Path) -> bytes:
        return ORCHESTRATOR._snapshot_secret_file(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_mode=0o640,
            label="API credential",
        )

    def test_secret_snapshots_accept_exact_api_and_token_contracts(self):
        api = self.write_secret("api-key", self.API_SECRET + b"\n", 0o640)
        token = self.write_secret("token", self.OPENWEBUI_TOKEN, 0o600)
        real_identity = ORCHESTRATOR._StableFileIdentity.from_stat

        def root_owned(value):
            return dataclasses.replace(real_identity(value), uid=0)

        with mock.patch.object(
            ORCHESTRATOR._StableFileIdentity,
            "from_stat",
            side_effect=root_owned,
        ):
            self.assertEqual(ORCHESTRATOR.snapshot_api_secret(api), self.API_SECRET)
        self.assertEqual(
            ORCHESTRATOR.snapshot_openwebui_token(token), self.OPENWEBUI_TOKEN
        )

    def test_secret_snapshot_rejects_symlink_mode_owner_and_multiple_lines(self):
        api = self.write_secret("api-key", self.API_SECRET, 0o640)
        symlink = self.root / "api-key-link"
        symlink.symlink_to(api)
        with self.assertRaises(ORCHESTRATOR.FullCampaignError):
            self.snapshot_api(symlink)

        api.chmod(0o600)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "owner, mode, or size"
        ):
            self.snapshot_api(api)
        api.chmod(0o640)
        real_identity = ORCHESTRATOR._StableFileIdentity.from_stat

        def wrong_owner(value):
            return dataclasses.replace(real_identity(value), uid=1)

        with self.assertRaisesRegex(ORCHESTRATOR.FullCampaignError, "owner"):
            with mock.patch.object(
                ORCHESTRATOR._StableFileIdentity,
                "from_stat",
                side_effect=wrong_owner,
            ):
                ORCHESTRATOR.snapshot_api_secret(api)

        token = self.write_secret(
            "token", self.OPENWEBUI_TOKEN + b"\nsecond-line", 0o600
        )
        with self.assertRaisesRegex(ORCHESTRATOR.FullCampaignError, "one bounded"):
            ORCHESTRATOR.snapshot_openwebui_token(token)

    def test_secret_snapshot_rejects_path_replacement_during_read(self):
        api = self.write_secret("api-key", self.API_SECRET, 0o640)
        replacement = self.write_secret(
            "replacement", b"replacement-api-secret-012345", 0o640
        )
        real_read = os.read
        replaced = False

        def replace_then_read(descriptor, maximum):
            nonlocal replaced
            if not replaced:
                os.replace(replacement, api)
                replaced = True
            return real_read(descriptor, maximum)

        with (
            mock.patch.object(ORCHESTRATOR.os, "read", side_effect=replace_then_read),
            self.assertRaisesRegex(
                ORCHESTRATOR.FullCampaignError, "changed while it was read"
            ),
        ):
            self.snapshot_api(api)

    def test_token_snapshot_rejects_mode_owner_and_hardlinks(self):
        token = self.write_secret("token", self.OPENWEBUI_TOKEN, 0o640)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "owner, mode, or size"
        ):
            ORCHESTRATOR.snapshot_openwebui_token(token)

        token.chmod(0o600)
        hardlink = self.root / "token-hardlink"
        os.link(token, hardlink)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "owner, mode, or size"
        ):
            ORCHESTRATOR.snapshot_openwebui_token(token)
        hardlink.unlink()

        real_identity = ORCHESTRATOR._StableFileIdentity.from_stat

        def wrong_owner(value):
            return dataclasses.replace(real_identity(value), uid=os.geteuid() + 1)

        with (
            mock.patch.object(
                ORCHESTRATOR._StableFileIdentity,
                "from_stat",
                side_effect=wrong_owner,
            ),
            self.assertRaisesRegex(ORCHESTRATOR.FullCampaignError, "owner"),
        ):
            ORCHESTRATOR.snapshot_openwebui_token(token)

    def test_secret_snapshot_rejects_fifo_without_blocking(self):
        fifo = self.root / "api-key-fifo"
        os.mkfifo(fifo, 0o640)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "owner, mode, or size"
        ):
            self.snapshot_api(fifo)

    def test_secret_owner_creates_private_masters_and_cleans_idempotently(self):
        owner = ORCHESTRATOR.CampaignSecretOwner.create(
            self.API_SECRET,
            self.OPENWEBUI_TOKEN,
            parent=self.root,
        )
        directory = owner.directory
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(owner.api_key_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(owner.openwebui_token_path.stat().st_mode), 0o600)
        self.assertEqual(owner.api_key_path.read_bytes(), self.API_SECRET)
        self.assertEqual(owner.openwebui_token_path.read_bytes(), self.OPENWEBUI_TOKEN)

        owner.close()
        owner.close()
        self.assertFalse(directory.exists())

    def test_secret_owner_removes_expected_masters_before_reporting_extra_entry(self):
        owner = ORCHESTRATOR.CampaignSecretOwner.create(
            self.API_SECRET,
            self.OPENWEBUI_TOKEN,
            parent=self.root,
        )
        extra = owner.directory / "unexpected"
        extra.write_bytes(b"unrelated")
        extra.chmod(0o600)

        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "changed before cleanup"
        ):
            owner.close()
        self.assertTrue(owner.closed)
        self.assertEqual(owner._directory_fd, -1)
        self.assertEqual(owner._parent_fd, -1)
        self.assertFalse(owner.api_key_path.exists())
        self.assertFalse(owner.openwebui_token_path.exists())
        self.assertTrue(extra.exists())
        owner.close()

        extra.unlink()
        owner.directory.rmdir()

    def test_secret_owner_deletes_modified_master_and_closes_all_descriptors(self):
        owner = ORCHESTRATOR.CampaignSecretOwner.create(
            self.API_SECRET,
            self.OPENWEBUI_TOKEN,
            parent=self.root,
        )
        directory = owner.directory
        owner.api_key_path.write_bytes(b"modified-api-secret-0123456789")
        owner.api_key_path.chmod(0o600)

        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "changed before cleanup"
        ):
            owner.close()
        self.assertTrue(owner.closed)
        self.assertEqual(owner._directory_fd, -1)
        self.assertEqual(owner._parent_fd, -1)
        self.assertFalse(directory.exists())

    def test_secret_owner_rejects_parent_replacement_during_creation(self):
        parent = self.root / "runtime"
        parent.mkdir(mode=0o700)
        moved = self.root / "runtime-old"
        real_write = ORCHESTRATOR._write_private_master
        calls = 0

        def write_then_replace(*args, **kwargs):
            nonlocal calls
            identity = real_write(*args, **kwargs)
            calls += 1
            if calls == 1:
                parent.rename(moved)
                parent.mkdir(mode=0o700)
            return identity

        with (
            mock.patch.object(
                ORCHESTRATOR,
                "_write_private_master",
                side_effect=write_then_replace,
            ),
            self.assertRaisesRegex(
                ORCHESTRATOR.FullCampaignError, "parent changed during"
            ),
        ):
            ORCHESTRATOR.CampaignSecretOwner.create(
                self.API_SECRET,
                self.OPENWEBUI_TOKEN,
                parent=parent,
            )
        self.assertEqual(list(moved.iterdir()), [])

    def test_composite_guard_rejects_json_escape_and_cross_chunk_secret(self):
        escaped_secret = b'escaped-"api\\secret-012345'
        token = self.OPENWEBUI_TOKEN
        guard = ORCHESTRATOR.CampaignSecretGuard((escaped_secret, token))
        encoded = json.dumps(
            {"credential": escaped_secret.decode("ascii")}, separators=(",", ":")
        ).encode("ascii")
        self.assertNotIn(escaped_secret, encoded)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "forbidden campaign cleartext"
        ):
            guard.reject(encoded, "escaped JSON")

        scanner = guard.scanner("chunked evidence")
        split = len(token) // 2
        scanner.feed(b"prefix-" + token[:split])
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "forbidden campaign cleartext"
        ):
            scanner.feed(token[split:] + b"-suffix")

    def test_composite_guard_duck_types_with_collector_writer(self):
        guard = ORCHESTRATOR.CampaignSecretGuard(
            (self.API_SECRET, self.OPENWEBUI_TOKEN)
        )
        output = self.root / "guarded.jsonl"
        writer = COLLECTOR.AtomicJsonlWriter(output, guard)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "forbidden campaign cleartext"
        ):
            writer.write_value({"escaped": self.API_SECRET.decode("ascii")})
        writer.abort_close()

    def test_composite_guard_scan_file_preserves_raw_chunk_overlap(self):
        guard = ORCHESTRATOR.CampaignSecretGuard(
            (self.API_SECRET, self.OPENWEBUI_TOKEN)
        )
        prefix_bytes = ORCHESTRATOR.SECRET_COPY_CHUNK_BYTES - len(self.API_SECRET) // 2
        evidence = self.root / "chunked-evidence.bin"
        evidence.write_bytes(b"x" * prefix_bytes + self.API_SECRET + b"suffix")
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "forbidden campaign cleartext"
        ):
            guard.scan_file(evidence, "chunked evidence file")

    def test_composite_guard_scans_multiline_json_and_ignores_non_json_shape(self):
        escaped_secret = b'escaped-"api\\secret-012345'
        guard = ORCHESTRATOR.CampaignSecretGuard((escaped_secret, self.OPENWEBUI_TOKEN))
        document = self.root / "evidence.json"
        document.write_text(
            json.dumps(
                {"nested": {"credential": escaped_secret.decode("ascii")}},
                indent=2,
            ),
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "forbidden campaign cleartext"
        ):
            guard.scan_file(document, "multiline JSON")

        empty = self.root / "empty.bin"
        empty.write_bytes(b"")
        guard.scan_file(empty, "empty binary")
        leading_t = self.root / "leading-t.bin"
        leading_t.write_bytes(b"this is not JSON")
        guard.scan_file(leading_t, "leading t binary")

    def test_composite_guard_fails_closed_on_oversized_json(self):
        guard = ORCHESTRATOR.CampaignSecretGuard(
            (self.API_SECRET, self.OPENWEBUI_TOKEN)
        )
        document = self.root / "oversized.json"
        document.write_text(json.dumps({"padding": "x" * 128}), encoding="ascii")
        with (
            mock.patch.object(ORCHESTRATOR, "SECRET_SCAN_JSON_MAX_BYTES", 64),
            self.assertRaisesRegex(
                ORCHESTRATOR.FullCampaignError, "semantic JSON scan bound"
            ),
        ):
            guard.scan_file(document, "oversized JSON")

    def test_composite_guard_bounds_flat_semantic_structures_before_enqueue(self):
        guard = ORCHESTRATOR.CampaignSecretGuard(
            (self.API_SECRET, self.OPENWEBUI_TOKEN)
        )
        oversized = [None] * (ORCHESTRATOR.SECRET_SCAN_MAX_NODES + 1)
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "semantic secret-scan node bound"
        ):
            guard.reject_json_value(oversized, "flat semantic array")

        raw = b"[" + (b"0," * ORCHESTRATOR.SECRET_SCAN_MAX_NODES) + b"0]"
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "semantic secret-scan node bound"
        ):
            guard.reject(raw, "flat raw JSON array")

    def test_campaign_lock_is_nonblocking_and_reusable_after_close(self):
        path = self.root / "full-campaign.lock"
        first = ORCHESTRATOR.CampaignLockOwner.acquire(path)
        try:
            with self.assertRaisesRegex(
                ORCHESTRATOR.FullCampaignError, "already holds the lock"
            ):
                ORCHESTRATOR.CampaignLockOwner.acquire(path)
        finally:
            first.close()
            first.close()

        second = ORCHESTRATOR.CampaignLockOwner.acquire(path)
        second.close()

    def test_campaign_lock_closes_descriptors_after_entry_replacement(self):
        path = self.root / "full-campaign.lock"
        owner = ORCHESTRATOR.CampaignLockOwner.acquire(path)
        replacement = self.root / "replacement.lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o600)
        os.replace(replacement, path)

        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "changed while held"
        ):
            owner.close()
        self.assertTrue(owner.closed)
        owner.close()
        with ORCHESTRATOR.CampaignLockOwner.acquire(path):
            pass

    def test_campaign_lock_rejects_replacement_during_flock(self):
        path = self.root / "full-campaign.lock"
        replacement = self.root / "replacement.lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o600)
        real_flock = ORCHESTRATOR.fcntl.flock
        replaced = False

        def lock_then_replace(descriptor, operation):
            nonlocal replaced
            result = real_flock(descriptor, operation)
            if operation & ORCHESTRATOR.fcntl.LOCK_EX and not replaced:
                os.replace(replacement, path)
                replaced = True
            return result

        with (
            mock.patch.object(
                ORCHESTRATOR.fcntl,
                "flock",
                side_effect=lock_then_replace,
            ),
            self.assertRaisesRegex(
                ORCHESTRATOR.FullCampaignError, "changed during acquisition"
            ),
        ):
            ORCHESTRATOR.CampaignLockOwner.acquire(path)

        with ORCHESTRATOR.CampaignLockOwner.acquire(path):
            pass


class FullCampaignOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_campaign(
        self, *, fail_phase=None, validator_fail=False, adapter_close_fail=False
    ):
        backend = FakeBackend(
            self.root,
            fail_phase=fail_phase,
            adapter_close_fail=adapter_close_fail,
        )
        final = self.root / "campaign"
        config = ORCHESTRATOR.CampaignConfig(
            final,
            os.getuid(),
            os.getgid(),
            BOOT_ID,
            ORCHESTRATOR.PidEpoch(NORMAL_GATEWAY, NORMAL_WORKER),
            operation_timeout_ns=3_000_000_000,
        )
        renderer = FakeRenderer(backend.calls)
        validator = FakeValidator(backend.calls, fail=validator_fail)
        result = ORCHESTRATOR.run_full_campaign(config, backend, renderer, validator)
        return backend, final, result

    def test_happy_path_uses_one_continuous_journal_and_publishes_last(self):
        backend, final, result = self.run_campaign()
        self.assertEqual(result, final)
        self.assertTrue(final.is_dir())
        self.assertEqual(
            [item for item in backend.calls if item.startswith("checkpoint:")],
            [f"checkpoint:{phase}" for phase in ORCHESTRATOR.PHASE_ORDER],
        )
        self.assertLess(
            backend.calls.index("checkpoint:resource_normal"),
            backend.calls.index("journal:arm_restart"),
        )
        self.assertLess(
            backend.calls.index("journal:arm_restart"),
            backend.calls.index("phase:post_header_failure"),
        )
        self.assertLess(
            backend.calls.index("journal:confirm_restart:2001:2002"),
            backend.calls.index("checkpoint:post_header_failure"),
        )
        self.assertEqual(backend.calls[-2:], ["render", "validate"])
        self.assertIn("journal:seal", backend.calls)
        self.assertTrue(backend.source.closed)
        self.assertFalse(backend.observer_open)

        records = [
            json.loads(line)
            for line in (final / "raw-session-results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(records[0]["record_type"], "header")
        self.assertEqual(records[-1]["record_type"], "run_end")
        quiet = [
            record
            for record in records
            if record["record_type"] == "lifecycle_quiet_check"
        ]
        self.assertEqual(len(quiet), 1)
        self.assertEqual(quiet[0]["journal_cursor"], "anchor-cursor")
        api_journal = [
            record
            for record in records
            if record["record_type"] == "api_journal_observation"
        ]
        self.assertEqual(len(api_journal), 1)
        self.assertEqual(api_journal[0]["observation_index"], 0)
        self.assertEqual(
            [
                record["probe"]
                for record in records
                if record["record_type"] == "lifecycle_probe"
            ],
            [
                "normal-segment-start",
                "post-header-restart-ready",
                "restart-segment-start",
                "final-service-ready",
            ],
        )
        failure_records = [
            record for record in records if record["phase"] == "post_header_failure"
        ]
        fatal_index = next(
            index
            for index, record in enumerate(failure_records)
            if record["record_type"] == "gateway_event"
            and record["event"]["event"] == "worker_fatal"
        )
        fault_index = next(
            index
            for index, record in enumerate(failure_records)
            if record["record_type"] == "fault_injection"
        )
        probe_index = next(
            index
            for index, record in enumerate(failure_records)
            if record["record_type"] == "lifecycle_probe"
        )
        recovery_action_index = next(
            index
            for index, record in enumerate(failure_records)
            if record["record_type"] == "browser_action"
            and record["case_id"] == "post-header-recovery"
        )
        recovery_gateway_index = next(
            index
            for index, record in enumerate(failure_records)
            if record["record_type"] == "gateway_event"
            and record["case_id"] == "post-header-recovery"
        )
        self.assertLess(fault_index, fatal_index)
        self.assertLess(fatal_index, probe_index)
        self.assertLess(probe_index, recovery_action_index)
        self.assertLess(probe_index, recovery_gateway_index)
        self.assertEqual(
            (final / "browser/openwebui-stop-before.png").read_bytes(),
            backend.stop_png.read_bytes(),
        )
        self.assertEqual(
            (final / "browser/post-header-failure.png").read_bytes(),
            backend.failure_png.read_bytes(),
        )

    def test_resource_result_requires_explicit_sampling_cases(self):
        result = types.SimpleNamespace(
            segment="normal",
            identity=Identity(
                "/system.slice/ullm-openai.service", 1001, 10001, 1002, 10002, 2
            ),
        )
        session = types.SimpleNamespace(counts={"lifecycle_probe": 1})
        with self.assertRaisesRegex(
            ORCHESTRATOR.FullCampaignError, "lacks sampling cases"
        ):
            ORCHESTRATOR._validate_resource_result(
                session,
                result,
                segment="normal",
                prior_probe_count=0,
            )

    def test_phase_failure_rolls_back_stage_and_work(self):
        with self.assertRaisesRegex(RuntimeError, "injected stop failure"):
            self.run_campaign(fail_phase="stop")
        self.assertFalse((self.root / "campaign").exists())
        self.assertFalse(any("incomplete" in path.name for path in self.root.iterdir()))
        self.assertFalse(any(".work-" in path.name for path in self.root.iterdir()))

    def test_validator_failure_never_publishes(self):
        with self.assertRaisesRegex(RuntimeError, "injected validator failure"):
            self.run_campaign(validator_fail=True)
        self.assertFalse((self.root / "campaign").exists())

    def test_cleanup_failure_does_not_mask_phase_error_and_all_cleanup_runs(self):
        backend = FakeBackend(
            self.root,
            fail_phase="post_header_failure",
            adapter_close_fail=True,
        )
        final = self.root / "campaign"
        config = ORCHESTRATOR.CampaignConfig(
            final,
            os.getuid(),
            os.getgid(),
            BOOT_ID,
            ORCHESTRATOR.PidEpoch(NORMAL_GATEWAY, NORMAL_WORKER),
            operation_timeout_ns=3_000_000_000,
        )
        with self.assertRaisesRegex(
            RuntimeError, "injected post_header_failure failure"
        ):
            ORCHESTRATOR.run_full_campaign(
                config,
                backend,
                FakeRenderer(backend.calls),
                FakeValidator(backend.calls),
            )
        self.assertIn("resource:close", backend.calls)
        self.assertIn("backend:close", backend.calls)
        self.assertIn("journal:abort", backend.calls)
        self.assertTrue(backend.source.closed)
        self.assertFalse(backend.observer_open)
        self.assertFalse(final.exists())
        self.assertFalse(
            any(path.name.startswith(".campaign") for path in self.root.iterdir())
        )

    def test_cli_is_fail_closed_until_production_backend_is_wired(self):
        self.assertEqual(ORCHESTRATOR.main([]), 2)
        self.assertEqual(ORCHESTRATOR.main(["--production-backend"]), 2)


if __name__ == "__main__":
    unittest.main()
