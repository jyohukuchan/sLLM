import base64
import dataclasses
import hashlib
import http.server
import importlib.util
import json
import os
import socket
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "tools" / "collect-sq8-openwebui-release.py"
VALIDATOR_PATH = ROOT / "tools" / "validate-sq8-openwebui-release.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load_module("collect_sq8_openwebui_release", COLLECTOR_PATH)
VALIDATOR = load_module("validate_sq8_openwebui_release_for_collector", VALIDATOR_PATH)

COMMIT = "a" * 40
WORKER_SHA256 = "b" * 64
SECRET = b"collector-test-secret-0123456789"
BOOT_ID = "5" * 32


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class FakeRuntime:
    def __init__(
        self,
        *,
        omit_release_at=None,
        secret_in_metric=False,
        final_idle_fatal=False,
        fatal_after_final_git=False,
        wrong_journal_pid=False,
        late_normal_boundary_event=False,
        final_git_callback=None,
        resource_identity_drift=False,
        negative_admission=False,
    ):
        self.now = 1_000_000_000_000
        self.started = False
        self.closed = False
        self.restarted = False
        self.pending_journal = []
        self.cursor_index = 0
        self.http_count = 0
        self.max_active = 0
        self.active = 0
        self.omit_release_at = omit_release_at
        self.secret_in_metric = secret_in_metric
        self.final_idle_fatal = final_idle_fatal
        self.final_idle_fatal_emitted = False
        self.fatal_after_final_git = fatal_after_final_git
        self.late_fatal_armed = False
        self.wrong_journal_pid = wrong_journal_pid
        self.late_normal_boundary_event = late_normal_boundary_event
        self.final_git_callback = final_git_callback
        self.git_identity_count = 0
        self.resource_identity_drift = resource_identity_drift
        self.negative_admission = negative_admission
        self.plans = []

    def now_ns(self):
        return self.now

    def wait_until(self, deadline_ns):
        self.now = max(self.now, deadline_ns)

    def wait_for_journal(self, deadline_ns):
        self.now = max(self.now, deadline_ns)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def boot_id(self):
        return BOOT_ID

    def lifecycle_probe(self):
        if self.restarted:
            identity = COLLECTOR.ProcessIdentity(
                "/system.slice/ullm-openai.service", 2200, 20000, 2201, 20001, 3
            )
        else:
            identity = COLLECTOR.ProcessIdentity(
                "/system.slice/ullm-openai.service", 1200, 10000, 1201, 10001, 2
            )
        self.now += 1_000_000
        return COLLECTOR.LifecycleProbe(self.now, True, 200, identity)

    def run_http(self, plan, emit):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active != 1:
            raise AssertionError("collector overlapped requests")
        self.http_count += 1
        self.plans.append(plan)
        try:
            connect = self._tick()
            request_fields = {
                "request_key": plan.request_key,
                "method": "POST",
                "target": plan.target,
                "headers": {
                    "content_type": "application/json",
                    "content_length": len(plan.body),
                    "authorization_mode": "valid_bearer",
                },
                "body_base64": base64.b64encode(plan.body).decode("ascii"),
                "body_sha256": hashlib.sha256(plan.body).hexdigest(),
                "body_bytes": len(plan.body),
                "connect_completed_monotonic_ns": connect,
                "write_started_monotonic_ns": connect + 1,
                "last_body_byte_sent_monotonic_ns": connect + 2,
            }
            emit("http_request", request_fields)
            start = self._tick()
            emit(
                "http_response_start",
                {
                    "request_key": plan.request_key,
                    "status": plan.expected_status,
                    "headers": [
                        [
                            "Content-Type",
                            "text/event-stream"
                            if plan.expected_status == 200
                            else "application/json",
                        ]
                    ],
                    "observed_monotonic_ns": start,
                },
            )
            completion_id = None
            if plan.expect_release:
                completion_id = f"chatcmpl-{self.http_count:04d}"
                body = (
                    b"data: "
                    + compact(
                        {
                            "id": completion_id,
                            "choices": [{"delta": {"content": "x"}}],
                        }
                    )
                    + b"\n\n"
                    + b"data: "
                    + compact(
                        {
                            "id": completion_id,
                            "choices": [],
                            "usage": {"completion_tokens": 2},
                        }
                    )
                    + b"\n\ndata: [DONE]\n\n"
                )
            else:
                expected_code = plan.expected_error_code or "invalid_request_error"
                body = compact(
                    {
                        "error": {
                            "message": "rejected",
                            "type": "invalid_request_error",
                            "param": "messages"
                            if expected_code == "context_length_exceeded"
                            else None,
                            "code": expected_code,
                        }
                    }
                )
            chunk_time = self._tick()
            emit(
                "http_body_chunk",
                {
                    "request_key": plan.request_key,
                    "chunk_index": 0,
                    "body_base64": base64.b64encode(body).decode("ascii"),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "body_bytes": len(body),
                    "observed_monotonic_ns": chunk_time,
                },
            )
            end_time = self._tick()
            emit(
                "http_response_end",
                {
                    "request_key": plan.request_key,
                    "outcome": "eof",
                    "error": None,
                    "body_bytes": len(body),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "observed_monotonic_ns": end_time,
                },
            )
            if plan.expect_release:
                self._queue_success_lifecycle(plan, completion_id)
            elif self.negative_admission:
                self._queue_negative_admission()
            return COLLECTOR.HttpObservation(
                status=plan.expected_status,
                completion_id=completion_id,
                outcome="eof",
            )
        finally:
            self.active -= 1

    def poll_journal(self):
        if (
            self.final_idle_fatal
            and self.closed
            and not self.pending_journal
            and not self.final_idle_fatal_emitted
        ):
            self.final_idle_fatal_emitted = True
            event = {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "worker_fatal",
                "observed_monotonic_ns": self._tick(),
                "request_id": None,
                "completion_id": None,
                "reason": "unexpected idle failure",
                "admit_to_fatal_ns": None,
            }
            self.cursor_index += 1
            self.pending_journal.append(
                compact(
                    {
                        "__CURSOR": f"cursor-{self.cursor_index:06d}",
                        "__MONOTONIC_TIMESTAMP": str(
                            event["observed_monotonic_ns"] // 1000
                        ),
                        "_BOOT_ID": BOOT_ID,
                        "_PID": "2200",
                        "_SYSTEMD_UNIT": "ullm-openai.service",
                        "PRIORITY": "3",
                        "MESSAGE": "INFO:     " + compact(event).decode("utf-8"),
                    }
                )
            )
        if (
            self.late_fatal_armed
            and not self.pending_journal
            and not self.final_idle_fatal_emitted
        ):
            self.final_idle_fatal_emitted = True
            event = {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "worker_fatal",
                "observed_monotonic_ns": self._tick(),
                "request_id": None,
                "completion_id": None,
                "reason": "late final failure",
                "admit_to_fatal_ns": None,
            }
            self.cursor_index += 1
            self.pending_journal.append(
                compact(
                    {
                        "__CURSOR": f"cursor-{self.cursor_index:06d}",
                        "__MONOTONIC_TIMESTAMP": str(
                            event["observed_monotonic_ns"] // 1000
                        ),
                        "_BOOT_ID": BOOT_ID,
                        "_PID": "2200",
                        "_SYSTEMD_UNIT": "ullm-openai.service",
                        "PRIORITY": "3",
                        "MESSAGE": "INFO:     " + compact(event).decode("utf-8"),
                    }
                )
            )
        if not self.pending_journal:
            return []
        return [self.pending_journal.pop(0)]

    def capture_metric(self, segment, boundary):
        self.now += 1_000_000
        if self.secret_in_metric:
            raw = compact({"gpu_data": [{"gpu": 2, "note": SECRET.decode()}]})
        else:
            raw = compact(
                {"gpu_data": [{"gpu": 2, "segment": segment, "boundary": boundary}]}
            )
        return COLLECTOR.MetricCapture(raw, self.now)

    def capture_resource(self):
        identity = self.lifecycle_probe().identity
        gateway = {
            "pid": identity.gateway_pid,
            "ppid": 1,
            "exe": "/usr/bin/python3.12",
            "starttime_ticks_before": identity.gateway_starttime_ticks,
            "starttime_ticks_after": identity.gateway_starttime_ticks,
            "vmrss_kb": 100000,
            "vmrss_bytes": 102400000,
            "threads": 8,
            "fd_count": 32,
            "children": [identity.worker_pid],
        }
        if self.resource_identity_drift:
            gateway["starttime_ticks_after"] += 1
        worker = {
            "pid": identity.worker_pid,
            "ppid": identity.gateway_pid,
            "exe": "/opt/ullm/bin/ullm-sq8-worker",
            "starttime_ticks_before": identity.worker_starttime_ticks,
            "starttime_ticks_after": identity.worker_starttime_ticks,
            "vmrss_kb": 200000,
            "vmrss_bytes": 204800000,
            "threads": 12,
            "fd_count": 24,
            "children": [],
        }
        return COLLECTOR.ResourceCapture(
            sample_monotonic_ns=self.now,
            systemd={
                "control_group_before": identity.control_group,
                "control_group_after": identity.control_group,
                "main_pid_before": identity.gateway_pid,
                "main_pid_after": identity.gateway_pid,
            },
            host={"memory_current_bytes": 1_000_000_000},
            gateway=gateway,
            worker=worker,
            gpu={
                "index": 2,
                "bdf": "0000:47:00.0",
                "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
                "kfd_gpu_id": 51545,
                "process_record_count": 1,
                "worker_pid": identity.worker_pid,
                "mem_usage": {"value": 20_000_000_000, "unit": "B"},
                "kfd_vram_bytes": 20_000_000_000,
                "unrelated_process_pids": [],
            },
        )

    def restart_hook(self):
        self.restarted = True
        self.now += 10_000_000
        if self.late_normal_boundary_event:
            event = {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "worker_fatal",
                "observed_monotonic_ns": self.now + 2_000_000,
                "request_id": None,
                "completion_id": None,
                "reason": "old gateway event beyond restart boundary",
                "admit_to_fatal_ns": None,
            }
            self.cursor_index += 1
            self.pending_journal.append(
                compact(
                    {
                        "__CURSOR": f"cursor-{self.cursor_index:06d}",
                        "__MONOTONIC_TIMESTAMP": str(
                            event["observed_monotonic_ns"] // 1000
                        ),
                        "_BOOT_ID": BOOT_ID,
                        "_PID": "1200",
                        "_SYSTEMD_UNIT": "ullm-openai.service",
                        "PRIORITY": "3",
                        "MESSAGE": "INFO:     " + compact(event).decode("utf-8"),
                    }
                )
            )
        return []

    def git_identity(self):
        self.git_identity_count += 1
        if self.git_identity_count == 2:
            if self.fatal_after_final_git:
                self.late_fatal_armed = True
            if self.final_git_callback is not None:
                self.final_git_callback()
        return COMMIT, ""

    def resource_header(self):
        return {
            "schema_version": COLLECTOR.RESOURCE_SCHEMA,
            "record_type": "header",
            "service_unit": "ullm-openai.service",
            "commands": dict(COLLECTOR.COMMANDS),
            "tools": {
                "systemd_major": 255,
                "systemd_version_line": "systemd 255 (synthetic)",
                "amd_smi_tool": "26.2.2+e1a6bc5663",
                "amd_smi_library": "26.2.2",
                "rocm": "7.2.1",
                "amd_smi_version_output": (
                    "AMDSMI Tool: 26.2.2+e1a6bc5663 | "
                    "AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1"
                ),
            },
            "probes": {
                "cgroup_fs_type": "cgroup2fs",
                "kfd_proc_present": True,
                "gpu_index": 2,
                "gpu_bdf": "0000:47:00.0",
                "gpu_uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
                "kfd_gpu_id": 51545,
            },
            "schedule": dict(COLLECTOR.RESOURCE_SCHEDULE),
        }

    def _tick(self, amount=1_000_000):
        self.now += amount
        return self.now

    def _queue_success_lifecycle(self, plan, completion_id):
        request_id = f"req-{self.http_count:04d}"
        base = self.now + 1_000_000
        events = [
            {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "request_admitted",
                "observed_monotonic_ns": base,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 8,
                "max_completion_tokens": 2,
            },
            {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "request_started",
                "observed_monotonic_ns": base + 1_000_000,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 8,
                "admit_to_start_ns": 10,
            },
            {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "request_first_token",
                "observed_monotonic_ns": base + 2_000_000,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "completion_tokens": 1,
            },
        ]
        if self.omit_release_at != self.http_count:
            events.append(
                {
                    "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                    "event": "request_released",
                    "observed_monotonic_ns": base + 3_000_000,
                    "request_id": request_id,
                    "completion_id": completion_id,
                    "stream": True,
                    "outcome": "length",
                    "cancel_reason": None,
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "reset_complete": True,
                    "admit_to_start_ns": 10,
                    "start_to_release_ns": 20,
                    "admit_to_release_ns": 30,
                }
            )
        for event in events:
            self.cursor_index += 1
            gateway_pid = 2200 if self.restarted else 1200
            if self.wrong_journal_pid and self.cursor_index == 1:
                gateway_pid += 99
            message = "INFO:     " + compact(event).decode("utf-8")
            record = {
                "__CURSOR": f"cursor-{self.cursor_index:06d}",
                "__MONOTONIC_TIMESTAMP": str(event["observed_monotonic_ns"] // 1000),
                "_BOOT_ID": BOOT_ID,
                "_PID": str(gateway_pid),
                "_SYSTEMD_UNIT": "ullm-openai.service",
                "PRIORITY": "6",
                "MESSAGE": message,
            }
            self.pending_journal.append(compact(record))

    def _queue_negative_admission(self):
        event = {
            "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
            "event": "request_admitted",
            "observed_monotonic_ns": self._tick(),
            "request_id": f"req-negative-{self.http_count:04d}",
            "completion_id": f"chatcmpl-negative-{self.http_count:04d}",
            "stream": True,
            "prompt_tokens": 8,
            "max_completion_tokens": 2,
        }
        self.cursor_index += 1
        gateway_pid = 2200 if self.restarted else 1200
        self.pending_journal.append(
            compact(
                {
                    "__CURSOR": f"cursor-{self.cursor_index:06d}",
                    "__MONOTONIC_TIMESTAMP": str(
                        event["observed_monotonic_ns"] // 1000
                    ),
                    "_BOOT_ID": BOOT_ID,
                    "_PID": str(gateway_pid),
                    "_SYSTEMD_UNIT": "ullm-openai.service",
                    "PRIORITY": "6",
                    "MESSAGE": "INFO:     " + compact(event).decode("utf-8"),
                }
            )
        )


class BufferedLifecycleClaims:
    """Synthetic stand-in for a campaign-owned continuous journal capture."""

    def __init__(self, runtime):
        self.runtime = runtime

    def consume(self, phase, case_id, expected_identity, completion_id=None):
        del phase, case_id
        result = []
        for raw in self.runtime.poll_journal():
            record = json.loads(raw)
            if int(record["_PID"]) != expected_identity.gateway_pid:
                raise COLLECTOR.CollectorError("buffered claim PID differs")
            event = COLLECTOR.decode_lifecycle_message(record["MESSAGE"])
            if event is not None:
                if (
                    completion_id is not None
                    and event.get("completion_id") != completion_id
                ):
                    raise COLLECTOR.CollectorError("buffered claim completion differs")
                result.append(event)
        return result

    def wait(self, deadline_ns):
        self.runtime.wait_for_journal(deadline_ns)

    def require_quiet(self, phase, case_id, expected_identity, deadline_ns):
        while True:
            if self.consume(phase, case_id, expected_identity):
                raise COLLECTOR.CollectorError("buffered negative claim is not quiet")
            if self.runtime.now_ns() >= deadline_ns:
                return
            self.wait(
                min(
                    deadline_ns,
                    self.runtime.now_ns() + COLLECTOR.JOURNAL_POLL_NS,
                )
            )


@dataclasses.dataclass
class FakeCampaignClaim:
    phase: str
    case_id: str
    fields: dict


class FakeCampaignCapture:
    def __init__(self, claims):
        self.claims = claims
        self.claim_calls = []
        self.quiet_calls = []

    def claim_completion_trace(self, completion_id, phase, case_id, deadline_ns):
        self.claim_calls.append((completion_id, phase, case_id, deadline_ns))
        return self.claims

    def wait_quiet(self, deadline_ns):
        self.quiet_calls.append(deadline_ns)
        return "quiet-cursor"


class RecordingSession:
    def __init__(self):
        self.records = []

    def append(self, record_type, phase, case_id, **fields):
        self.records.append((record_type, phase, case_id, fields))


class CampaignResourceLifecycleClaimsTests(unittest.TestCase):
    def setUp(self):
        self.identity = COLLECTOR.ProcessIdentity(
            "/system.slice/ullm-openai.service", 1200, 10000, 1201, 10001, 2
        )
        self.now = 1_000_000_000
        self.waited = []

    @staticmethod
    def claim(event_name, *, pid=1200, completion_id="chatcmpl-1"):
        return FakeCampaignClaim(
            "resource_normal",
            "normal-measured-001",
            {
                "journal_cursor": f"cursor-{event_name}",
                "journal_monotonic_usec": 1000,
                "journal_pid": pid,
                "message": event_name,
                "message_sha256": hashlib.sha256(event_name.encode()).hexdigest(),
                "event": {
                    "event": event_name,
                    "completion_id": completion_id,
                },
            },
        )

    def adapter(self, claims):
        capture = FakeCampaignCapture(claims)
        session = RecordingSession()
        adapter = COLLECTOR.CampaignResourceLifecycleClaims(
            capture,
            session,
            lambda: self.now,
            self.waited.append,
        )
        return adapter, capture, session

    def test_claims_one_completion_and_appends_only_validated_events(self):
        claims = [
            self.claim("request_admitted"),
            self.claim("request_released"),
        ]
        adapter, capture, session = self.adapter(claims)
        events = adapter.consume(
            "resource_normal",
            "normal-measured-001",
            self.identity,
            "chatcmpl-1",
        )
        self.assertEqual(
            [event["event"] for event in events],
            ["request_admitted", "request_released"],
        )
        self.assertEqual(len(session.records), 2)
        self.assertEqual(
            capture.claim_calls,
            [
                (
                    "chatcmpl-1",
                    "resource_normal",
                    "normal-measured-001",
                    self.now + COLLECTOR.RELEASE_TIMEOUT_NS,
                )
            ],
        )
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "claimed twice"):
            adapter.consume(
                "resource_normal",
                "normal-measured-001",
                self.identity,
                "chatcmpl-1",
            )

    def test_rejects_entire_trace_before_append_on_pid_or_completion_drift(self):
        mutations = [
            [self.claim("request_admitted"), self.claim("request_released", pid=2200)],
            [
                self.claim("request_admitted"),
                self.claim("request_released", completion_id="chatcmpl-other"),
            ],
        ]
        for claims in mutations:
            with self.subTest(claims=claims):
                adapter, _capture, session = self.adapter(claims)
                with self.assertRaisesRegex(
                    COLLECTOR.CollectorError, "process or completion"
                ):
                    adapter.consume(
                        "resource_normal",
                        "normal-measured-001",
                        self.identity,
                        "chatcmpl-1",
                    )
                self.assertEqual(session.records, [])

    def test_requires_completion_and_delegates_wait_and_quiet(self):
        adapter, capture, _session = self.adapter([self.claim("request_admitted")])
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "completion ID"):
            adapter.consume("resource_normal", "normal-measured-001", self.identity)
        adapter.wait(1_100_000_000)
        adapter.require_quiet(
            "resource_normal",
            "negative-after-025-context_overflow_1",
            self.identity,
            1_200_000_000,
        )
        self.assertEqual(self.waited, [1_100_000_000])
        self.assertEqual(capture.quiet_calls, [1_200_000_000])


class CollectorTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "bundle"
        self.guard = COLLECTOR.SecretGuard(SECRET)
        self.config = self._config()

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def _config(self):
        phase_artifacts = {}
        for relative in sorted(COLLECTOR.PHASE_ARTIFACT_PATHS):
            if relative.endswith(".png"):
                raw = b"\x89PNG\r\n\x1a\nsynthetic"
            elif relative.endswith(".md"):
                raw = b"synthetic phase-1 evidence\n"
            else:
                raw = compact({"schema_version": "synthetic.incomplete.v1"})
            phase_artifacts[relative] = self._write(
                "synthetic-phase-artifacts/" + relative.replace("/", "_"), raw
            )
        template = {
            "model": "Qwen3-14B-SQ8",
            "messages": [{"role": "user", "content": "fixture"}],
        }
        input_path = self._write("input.json", compact({"fixture": "resource"}))
        fixture_raw = compact(template)
        collector_raw = COLLECTOR_PATH.read_bytes()
        client_raw = (ROOT / "tools" / "sq8-openwebui-http-client.py").read_bytes()
        inputs = [
            COLLECTOR.InputFile("collector/config.json", input_path, snapshot=b"{}"),
            COLLECTOR.InputFile(
                COLLECTOR.RESOURCE_FIXTURE_INPUT_PATH,
                input_path,
                snapshot=fixture_raw,
            ),
            COLLECTOR.InputFile("fixtures/resource.json", input_path),
            COLLECTOR.InputFile(
                "tools/collect-sq8-openwebui-release.py",
                COLLECTOR_PATH,
                snapshot=collector_raw,
            ),
            COLLECTOR.InputFile(
                COLLECTOR.HTTP_CLIENT_INPUT_PATH,
                ROOT / "tools" / "sq8-openwebui-http-client.py",
                snapshot=client_raw,
            ),
        ]
        inputs.sort(key=lambda item: item.path.encode("utf-8"))

        def overflow(marker):
            return compact(
                {
                    "model": template["model"],
                    "messages": [
                        {"role": "user", "content": marker + (" overflow" * 5000)}
                    ],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 2,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 0,
                }
            )

        return COLLECTOR.CollectorConfig(
            run_id="synthetic-phase1-run",
            identities={
                "openwebui": {
                    "version": "0.9.4",
                    "source_revision": "revision",
                    "base_image_digest": "sha256:" + "1" * 64,
                    "base_image_id": "sha256:" + "2" * 64,
                    "derived_image_id": "sha256:" + "3" * 64,
                    "Dockerfile_sha256": "4" * 64,
                    "patch_sha256": "5" * 64,
                    "patched_middleware_sha256": "6" * 64,
                },
                "docker_network_id": "9" * 64,
                "gateway_source_sha256": "7" * 64,
                "worker_source_sha256": "8" * 64,
                "worker_binary_sha256": WORKER_SHA256,
            },
            input_files=tuple(inputs),
            phase_artifacts=phase_artifacts,
            target="/v1/chat/completions",
            resource_body_template=template,
            negative_cases=(
                COLLECTOR.NegativeCase(25, "context_overflow_1", overflow("one"), 400),
                COLLECTOR.NegativeCase(50, "malformed_json", b"{", 400),
                COLLECTOR.NegativeCase(75, "context_overflow_2", overflow("two"), 400),
            ),
            restart_command=("unused",),
            ready_url=COLLECTOR.HTTP_READY_URL,
            amd_smi=COLLECTOR.AMD_SMI_BIN,
        )

    def _collector(self, runtime):
        return COLLECTOR.Phase1Collector(
            self.config,
            self.output,
            self.root,
            COMMIT,
            WORKER_SHA256,
            self.guard,
            runtime,
        )

    def _resource_component(self, runtime, name="component"):
        output = self.root / name
        output.mkdir()
        runtime.start()
        session = COLLECTOR.SessionWriter(output / "session.jsonl", self.guard)
        resource = COLLECTOR.AtomicJsonlWriter(output / "resources.jsonl", self.guard)
        journal = COLLECTOR.JournalState(
            boot_id=runtime.boot_id(),
            raw_writer=COLLECTOR.AtomicJsonlWriter(
                output / "journal.jsonl", self.guard
            ),
            session=session,
        )
        claims = COLLECTOR.Phase1ResourceLifecycleClaims(runtime, journal)
        component = COLLECTOR.ResourceSegmentCollector(
            COLLECTOR.ResourceSegmentConfig.from_collector_config(self.config),
            output,
            self.guard,
            runtime,
            session,
            resource,
            claims,
        )
        return output, session, resource, journal, component

    @staticmethod
    def _close_resource_component(session, resource, journal):
        session.writer.abort_close()
        resource.abort_close()
        journal.raw_writer.abort_close()

    def test_readiness_url_is_the_deployed_readyz_endpoint(self):
        self.assertEqual(COLLECTOR.HTTP_READY_URL, "http://172.20.0.1:8000/readyz")
        document = self._valid_config_document()
        document["ready_url"] = "http://172.20.0.1:8000/ready"
        path = self._write("wrong-ready-config.json", compact(document))
        with self.assertRaisesRegex(
            COLLECTOR.CollectorError, "fixed bridge readiness URL"
        ):
            COLLECTOR.load_config(path)

    def test_synthetic_phase1_bundle_passes_independent_validator(self):
        runtime = FakeRuntime()
        result = self._collector(runtime).run()
        self.assertTrue(result["phase1_collected"])
        self.assertEqual(result["release_status"], "incomplete")
        self.assertEqual(runtime.max_active, 1)
        self.assertTrue(runtime.closed)
        report = VALIDATOR.validate_phase1(
            self.output,
            expected_commit=COMMIT,
            expected_worker_binary_sha256=WORKER_SHA256,
        )
        self.assertTrue(report["phase1_validated"])
        self.assertEqual(report["release_status"], "incomplete")
        self.assertEqual(report["raw_counts"]["resource_samples"], 610)
        self.assertEqual(report["raw_counts"]["gpu_metrics"], 4)
        self.assertFalse((self.output / "release-validation.json").exists())
        self.assertEqual(list(self.output.rglob("*.incomplete")), [])
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET, path.read_bytes())

    def test_resource_component_collects_normal_segment_only(self):
        runtime = FakeRuntime()
        output, session, resource, journal, component = self._resource_component(
            runtime, "normal-component"
        )
        self.config.resource_body_template["messages"][0]["content"] = (
            "mutated-after-component-construction"
        )
        try:
            result = component.collect_normal()
            self.assertEqual(result.segment, "normal")
            self.assertEqual(result.identity.gateway_pid, 1200)
            self.assertEqual(result.warmup_requests, 10)
            self.assertEqual(result.measured_requests, 100)
            self.assertEqual(result.negative_requests, 3)
            self.assertEqual(result.resource_samples, 505)
            self.assertEqual(result.gpu_metrics, 2)
            expected_sampling_cases = tuple(
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
            self.assertEqual(result.sampling_cases, expected_sampling_cases)
            self.assertTrue(
                all(
                    type(case["temperature"]) is float and type(case["top_p"]) is float
                    for case in result.sampling_cases
                )
            )
            sampled_plans = {
                plan.request_index: json.loads(plan.body)
                for plan in runtime.plans
                if plan.case_id.startswith("normal-measured-")
                and plan.request_index in COLLECTOR.SAMPLED_NORMAL_INDICES
            }
            self.assertEqual(tuple(sorted(sampled_plans)), tuple(range(5, 101, 5)))
            self.assertEqual(
                tuple(
                    (
                        sampled_plans[index]["temperature"],
                        sampled_plans[index]["top_p"],
                        sampled_plans[index]["seed"],
                    )
                    for index in sorted(sampled_plans)
                ),
                tuple((0.6, 0.95, index) for index in range(5, 101, 5)),
            )
            self.assertEqual(runtime.http_count, 113)
            self.assertEqual(runtime.max_active, 1)
            self.assertFalse(runtime.restarted)
            self.assertFalse(runtime.closed)
            self.assertEqual(resource.line_count, 507)
            self.assertEqual(session.counts["lifecycle_probe"], 1)
            self.assertEqual(session.counts["http_request"], 113)
            self.assertEqual(session.counts["gateway_event"], 440)
            self.assertEqual(journal.raw_writer.line_count, 440)
            self.assertEqual(runtime.plans[34].case_id, "normal-measured-025")
            self.assertEqual(
                json.loads(runtime.plans[0].body)["messages"],
                [{"role": "user", "content": "fixture"}],
            )
            self.assertEqual(
                runtime.plans[35].case_id,
                "negative-after-025-context_overflow_1",
            )
            self.assertEqual(runtime.plans[36].case_id, "normal-measured-026")
            self.assertEqual(
                {
                    path.name
                    for path in output.iterdir()
                    if path.name.startswith("amd-smi-metric-")
                },
                {
                    "amd-smi-metric-normal-before.json",
                    "amd-smi-metric-normal-after.json",
                },
            )
            self.assertFalse(hasattr(component.config, "phase_artifacts"))
            with self.assertRaisesRegex(COLLECTOR.CollectorError, "duplicated"):
                component.collect_normal()
            self.assertEqual(runtime.http_count, 113)
        finally:
            self._close_resource_component(session, resource, journal)

    def test_resource_component_collects_restart_segment_only(self):
        runtime = FakeRuntime()
        runtime.restarted = True
        output, session, resource, journal, component = self._resource_component(
            runtime, "restart-component"
        )
        normal_identity = COLLECTOR.ProcessIdentity(
            "/system.slice/ullm-openai.service", 1200, 10000, 1201, 10001, 2
        )
        restart_identity = COLLECTOR.ProcessIdentity(
            "/system.slice/ullm-openai.service", 2200, 20000, 2201, 20001, 3
        )
        try:
            result = component.collect_restart(
                normal_identity, expected_identity=restart_identity
            )
            self.assertEqual(result.segment, "restart")
            self.assertEqual(result.identity, restart_identity)
            self.assertEqual(result.warmup_requests, 10)
            self.assertEqual(result.measured_requests, 20)
            self.assertEqual(result.negative_requests, 0)
            self.assertEqual(result.resource_samples, 105)
            self.assertEqual(result.sampling_cases, ())
            self.assertEqual(runtime.http_count, 30)
            self.assertEqual(runtime.max_active, 1)
            self.assertFalse(runtime.closed)
            self.assertEqual(resource.line_count, 107)
            self.assertEqual(session.counts["http_request"], 30)
            self.assertEqual(session.counts["gateway_event"], 120)
            self.assertEqual(journal.raw_writer.line_count, 120)
            session.writer.file.sync()
            records = [
                json.loads(line)
                for line in session.writer.file.incomplete_path.read_text().splitlines()
            ]
            self.assertEqual(
                {record["phase"] for record in records}, {"resource_restart"}
            )
            self.assertEqual(
                {
                    path.name
                    for path in output.iterdir()
                    if path.name.startswith("amd-smi-metric-")
                },
                {
                    "amd-smi-metric-restart-before.json",
                    "amd-smi-metric-restart-after.json",
                },
            )
        finally:
            self._close_resource_component(session, resource, journal)

    def test_resource_component_accepts_campaign_owned_buffered_claims(self):
        runtime = FakeRuntime()
        runtime.restarted = True
        output, session, resource, journal, original = self._resource_component(
            runtime, "buffered-claims-component"
        )
        component = COLLECTOR.ResourceSegmentCollector(
            original.config,
            output,
            self.guard,
            runtime,
            session,
            resource,
            BufferedLifecycleClaims(runtime),
        )
        normal_identity = COLLECTOR.ProcessIdentity(
            "/system.slice/ullm-openai.service", 1200, 10000, 1201, 10001, 2
        )
        try:
            result = component.collect_restart(normal_identity)
            self.assertEqual(result.identity.gateway_pid, 2200)
            self.assertEqual(result.sampling_cases, ())
            self.assertEqual(runtime.http_count, 30)
            self.assertEqual(resource.line_count, 107)
            self.assertEqual(journal.raw_writer.line_count, 0)
            self.assertEqual(session.counts["gateway_event"], 0)
        finally:
            self._close_resource_component(session, resource, journal)

    def test_resource_component_rejects_start_and_sample_identity_drift(self):
        runtime = FakeRuntime()
        _, session, resource, journal, component = self._resource_component(
            runtime, "start-drift-component"
        )
        wrong_identity = COLLECTOR.ProcessIdentity(
            "/system.slice/ullm-openai.service", 999, 10000, 1201, 10001, 2
        )
        try:
            with self.assertRaisesRegex(COLLECTOR.CollectorError, "campaign epoch"):
                component.collect_normal(expected_identity=wrong_identity)
            self.assertEqual(runtime.http_count, 0)
        finally:
            self._close_resource_component(session, resource, journal)

        runtime = FakeRuntime(resource_identity_drift=True)
        _, session, resource, journal, component = self._resource_component(
            runtime, "sample-drift-component"
        )
        try:
            with self.assertRaisesRegex(
                COLLECTOR.CollectorError, "process identity changed"
            ):
                component.collect_normal()
            self.assertEqual(runtime.http_count, 10)
        finally:
            self._close_resource_component(session, resource, journal)

    def test_resource_component_negative_claim_stops_before_next_request(self):
        runtime = FakeRuntime(negative_admission=True)
        _, session, resource, journal, component = self._resource_component(
            runtime, "negative-claim-component"
        )
        try:
            with self.assertRaisesRegex(
                COLLECTOR.CollectorError, "negative request produced"
            ):
                component.collect_normal()
            self.assertEqual(runtime.http_count, 36)
            self.assertEqual(
                runtime.plans[-1].case_id,
                "negative-after-025-context_overflow_1",
            )
        finally:
            self._close_resource_component(session, resource, journal)

    def test_resource_component_rejects_schedule_body_and_template_mutations(self):
        runtime = FakeRuntime()
        output, session, resource, journal, component = self._resource_component(
            runtime, "invalid-config-component"
        )
        base = component.config
        mutations = [
            dataclasses.replace(
                base, negative_cases=tuple(reversed(base.negative_cases))
            ),
            dataclasses.replace(
                base,
                negative_cases=tuple(
                    dataclasses.replace(item, body=b"{}")
                    if item.name == "malformed_json"
                    else item
                    for item in base.negative_cases
                ),
            ),
            dataclasses.replace(
                base,
                negative_cases=tuple(
                    dataclasses.replace(item, body=compact({"wrong": True}))
                    if item.name == "context_overflow_1"
                    else item
                    for item in base.negative_cases
                ),
            ),
            dataclasses.replace(
                base,
                resource_body_template={
                    "model": "Qwen3-14B-SQ8",
                    "messages": [{"role": "tool", "content": "fixture"}],
                },
            ),
        ]
        try:
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    with self.assertRaises(COLLECTOR.CollectorError):
                        COLLECTOR.ResourceSegmentCollector(
                            mutation,
                            output,
                            self.guard,
                            runtime,
                            session,
                            resource,
                            component.lifecycle_claims,
                        )
            self.assertEqual(runtime.http_count, 0)
        finally:
            self._close_resource_component(session, resource, journal)

    def test_missing_release_times_out_before_next_request_and_keeps_incomplete(self):
        runtime = FakeRuntime(omit_release_at=1)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "timed out waiting"):
            self._collector(runtime).run()
        self.assertEqual(runtime.http_count, 1)
        self.assertTrue(runtime.closed)
        self.assertTrue((self.output / "raw-session-results.jsonl.incomplete").exists())
        self.assertFalse((self.output / "release-matrix.json").exists())

    def test_secret_in_metric_fails_without_publishing_raw_files(self):
        runtime = FakeRuntime(secret_in_metric=True)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "API credential"):
            self._collector(runtime).run()
        self.assertFalse((self.output / "soak-resources.raw.jsonl").exists())
        self.assertTrue((self.output / "soak-resources.raw.jsonl.incomplete").exists())

    def test_replaced_phase_artifact_is_not_scanned_or_published_by_path(self):
        identities = {
            relative: (path.stat().st_dev, path.stat().st_ino)
            for relative, path in self.config.phase_artifacts.items()
        }
        config = dataclasses.replace(
            self.config,
            phase_artifact_identities=identities,
        )
        environment = config.phase_artifacts["environment.json"]
        replacement = environment.with_name("replacement-environment.json")
        replacement.write_bytes(compact({"secret": SECRET.decode()}))
        replacement.replace(environment)
        collector = COLLECTOR.Phase1Collector(
            config,
            self.output,
            self.root,
            COMMIT,
            WORKER_SHA256,
            self.guard,
            FakeRuntime(),
        )
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "identity changed"):
            collector.run()
        self.assertFalse((self.output / "environment.json").exists())
        incomplete = self.output / "environment.json.incomplete"
        self.assertTrue(incomplete.exists())
        self.assertEqual(incomplete.stat().st_size, 0)

    def test_input_file_identity_is_bound_to_single_fd_inspection(self):
        source = self._write("bound-input.json", b"{}")
        metadata = source.stat()
        item = COLLECTOR.InputFile(
            "fixtures/bound-input.json",
            source,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )
        replacement = self._write("bound-input-replacement.json", b'{"changed":true}')
        replacement.replace(source)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "identity changed"):
            COLLECTOR.inspect_input_file(item, self.guard)

    def test_final_idle_worker_fatal_prevents_bundle_seal(self):
        runtime = FakeRuntime(final_idle_fatal=True)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "final journal drain"):
            self._collector(runtime).run()
        self.assertFalse((self.output / "raw-session-results.jsonl").exists())
        self.assertTrue((self.output / "raw-session-results.jsonl.incomplete").exists())

    def test_fatal_after_heavy_final_work_is_drained_before_run_end(self):
        runtime = FakeRuntime(fatal_after_final_git=True)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "final journal drain"):
            self._collector(runtime).run()
        self.assertFalse((self.output / "raw-session-results.jsonl").exists())
        self.assertTrue((self.output / "raw-session-results.jsonl.incomplete").exists())

    def test_fatal_during_matrix_sealing_invalidates_bundle_after_final_probe(self):
        runtime = FakeRuntime()
        collector = self._collector(runtime)
        write_matrix = collector._write_matrix

        def write_matrix_and_arm_fatal():
            write_matrix()
            runtime.late_fatal_armed = True

        collector._write_matrix = write_matrix_and_arm_fatal
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "post-seal final drain"):
            collector.run()
        self.assertFalse((self.output / "SHA256SUMS").exists())
        self.assertTrue((self.output / "SHA256SUMS.incomplete").exists())

    def test_abrupt_exit_during_publish_probe_cannot_leave_complete_checksum(self):
        collector = self._collector(FakeRuntime())

        def interrupt_publish_probe():
            raise SystemExit("simulated abrupt collector exit")

        collector._verify_post_seal_service_state = interrupt_publish_probe
        with self.assertRaises(SystemExit):
            collector.run()
        self.assertFalse((self.output / "SHA256SUMS").exists())
        self.assertTrue((self.output / "SHA256SUMS.incomplete").exists())

    def test_wrong_gateway_journal_pid_fails_closed(self):
        runtime = FakeRuntime(wrong_journal_pid=True)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "journal PID differs"):
            self._collector(runtime).run()
        self.assertEqual(runtime.http_count, 1)
        self.assertTrue((self.output / "raw-session-results.jsonl.incomplete").exists())

    def test_old_gateway_event_cannot_exceed_restart_ready_boundary(self):
        runtime = FakeRuntime(late_normal_boundary_event=True)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "identity epoch"):
            self._collector(runtime).run()
        self.assertFalse((self.output / "raw-session-results.jsonl").exists())

    def test_staged_artifact_change_is_rejected_before_matrix(self):
        def replace_summary():
            (self.output / "summary.md").write_bytes(b"changed after staging\n")

        runtime = FakeRuntime(final_git_callback=replace_summary)
        with self.assertRaisesRegex(
            COLLECTOR.CollectorError, "staged identity or content"
        ):
            self._collector(runtime).run()
        self.assertFalse((self.output / "release-matrix.json").exists())

    def test_hook_jsonl_record_limit_prevents_object_amplification(self):
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "record-count limit"):
            list(
                COLLECTOR.stream_bounded_jsonl_command(
                    (
                        sys.executable,
                        "-c",
                        (
                            "import sys;"
                            f"[sys.stdout.write('{{}}\\n') for _ in range({COLLECTOR.MAX_HOOK_RECORDS + 1})];"
                            "sys.stdout.flush()"
                        ),
                    ),
                    "hook",
                    maximum_records=COLLECTOR.MAX_HOOK_RECORDS,
                )
            )

    def test_private_credential_snapshot_rejects_mode_and_symlink(self):
        credential = self._write("credential", SECRET + b"\n")
        credential.chmod(0o644)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "permissions"):
            COLLECTOR.SecretGuard.snapshot_from_file(credential)
        credential.chmod(0o600)
        link = self.root / "credential-link"
        link.symlink_to(credential)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "without following"):
            COLLECTOR.SecretGuard.snapshot_from_file(link)

    def test_collector_config_snapshot_rejects_final_symlink(self):
        config = self._write("real-config.json", compact(self._valid_config_document()))
        link = self.root / "config-link.json"
        link.symlink_to(config)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "without following"):
            COLLECTOR.load_config(link)

    def test_runtime_snapshots_build_only_fixed_docker_command_and_cleanup(self):
        self.root.chmod(0o700)
        snapshots = COLLECTOR.RuntimeSnapshots.create(
            b"print('client')\n",
            SECRET,
            parent=self.root,
        )
        try:
            command = COLLECTOR.build_http_client_command(self.config, snapshots)
            self.assertEqual(command[0], COLLECTOR.DOCKER_BIN)
            self.assertIn("--interactive", command)
            self.assertIn("--network=open-webui-network", command)
            self.assertIn("--user=1000:1000", command)
            self.assertIn(
                self.config.identities["openwebui"]["derived_image_id"], command
            )
            self.assertIn(COLLECTOR.HTTP_BASE_URL, command)
            self.assertEqual(snapshots.directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(snapshots.credential_path.stat().st_mode & 0o777, 0o600)
            snapshots.unlink_credential()
            self.assertFalse(snapshots.credential_path.exists())
        finally:
            directory = snapshots.directory
            snapshots.close()
        self.assertFalse(directory.exists())

    def test_system_runtime_config_is_deeply_snapshotted_and_immutable(self):
        identities = json.loads(json.dumps(self.config.identities))
        expected_image = identities["openwebui"]["derived_image_id"]
        runtime_config = COLLECTOR.SystemRuntimeConfig.for_full_campaign(
            identities=identities,
            amd_smi=COLLECTOR.AMD_SMI_BIN,
        )
        identities["openwebui"]["derived_image_id"] = "sha256:" + "f" * 64
        identities["worker_binary_sha256"] = "e" * 64
        self.assertEqual(
            runtime_config.identities["openwebui"]["derived_image_id"],
            expected_image,
        )
        self.assertEqual(
            runtime_config.identities["worker_binary_sha256"], WORKER_SHA256
        )
        nested = runtime_config.identities["openwebui"]
        with self.assertRaises(TypeError):
            nested["derived_image_id"] = "sha256:" + "d" * 64
        with self.assertRaises(dataclasses.FrozenInstanceError):
            runtime_config.amd_smi = "/tmp/changed"
        self.assertIsNone(runtime_config.restart_command)

    def test_system_runtime_config_rejects_invalid_runtime_identities(self):
        mutations = (
            ("missing worker hash", lambda value: value.pop("worker_source_sha256")),
            (
                "invalid image",
                lambda value: value["openwebui"].__setitem__(
                    "derived_image_id", "sha256:not-a-digest"
                ),
            ),
            (
                "invalid network",
                lambda value: value.__setitem__("docker_network_id", "0" * 63),
            ),
            (
                "invalid source hash",
                lambda value: value.__setitem__("gateway_source_sha256", "A" * 64),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                identities = json.loads(json.dumps(self.config.identities))
                mutate(identities)
                with self.assertRaises(COLLECTOR.CollectorError):
                    COLLECTOR.SystemRuntimeConfig.validated(
                        identities=identities,
                        amd_smi=COLLECTOR.AMD_SMI_BIN,
                    )
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "fixed executable"):
            COLLECTOR.SystemRuntimeConfig.validated(
                identities=self.config.identities,
                amd_smi="/tmp/amd-smi",
            )

    def test_system_runtime_extracts_only_runtime_fields_from_collector_config(self):
        class Snapshots:
            client_path = Path("/tmp/sq8-client.py")
            credential_path = Path("/tmp/sq8-credential")

            @staticmethod
            def unlink_credential():
                raise AssertionError("runtime was not started")

        runtime = COLLECTOR.SystemRuntime(
            self.config, self.root, self.guard, Snapshots()
        )
        self.assertIsInstance(runtime.config, COLLECTOR.SystemRuntimeConfig)
        self.assertFalse(hasattr(runtime.config, "phase_artifacts"))
        self.assertFalse(hasattr(runtime.config, "input_files"))
        self.assertEqual(runtime.config.restart_command, ("unused",))
        self.assertTrue(runtime.capture_journal)
        runtime.close()

    def test_http_command_accepts_minimal_identity_only_config(self):
        class IdentityOnly:
            def __init__(self, identities):
                self.identities = identities

        class Snapshots:
            client_path = Path("/tmp/sq8-client.py")
            credential_path = Path("/tmp/sq8-credential")

        command = COLLECTOR.build_http_client_command(
            IdentityOnly(self.config.identities), Snapshots()
        )
        self.assertIn(self.config.identities["openwebui"]["derived_image_id"], command)

    def test_system_runtime_external_journal_mode_keeps_http_and_disables_polling(self):
        class Snapshots:
            client_path = Path("/tmp/sq8-client.py")
            credential_path = Path("/tmp/sq8-credential")
            unlinked = False

            def unlink_credential(self):
                self.unlinked = True

        class FakeHttp:
            def __init__(self):
                self.started = False
                self.closed = False

            def start(self):
                self.started = True

            def close(self):
                self.closed = True

            @staticmethod
            def request(plan, emit):
                return (plan, emit)

        snapshots = Snapshots()
        runtime_config = COLLECTOR.SystemRuntimeConfig.for_full_campaign(
            identities=self.config.identities,
            amd_smi=COLLECTOR.AMD_SMI_BIN,
        )
        runtime = COLLECTOR.SystemRuntime(
            runtime_config,
            self.root,
            self.guard,
            snapshots,
            capture_journal=False,
        )
        http = FakeHttp()
        runtime.http = http
        with (
            mock.patch.object(runtime, "_validate_docker_identity"),
            mock.patch.object(
                COLLECTOR.JournalSource,
                "__init__",
                side_effect=AssertionError("journal source must not be constructed"),
            ),
        ):
            runtime.start()
        self.assertTrue(http.started)
        self.assertTrue(snapshots.unlinked)
        marker = object()
        observation = runtime.run_http(marker, lambda _event, _fields: None)
        self.assertIs(observation[0], marker)
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "disabled"):
            runtime.poll_journal()
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "disabled"):
            runtime.wait_for_journal(runtime.now_ns())
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "restart command"):
            runtime.restart_hook()
        runtime.close()
        self.assertTrue(http.closed)

    def test_system_runtime_start_failure_closes_http_and_journal(self):
        class Snapshots:
            client_path = Path("/tmp/sq8-client.py")
            credential_path = Path("/tmp/sq8-credential")

            @staticmethod
            def unlink_credential():
                raise AssertionError("failed start must not unlink the credential")

        class FakeJournal:
            def __init__(self, _boot_id):
                self.started = False
                self.closed = False

            def start(self):
                self.started = True

            def close(self):
                self.closed = True

        class FailingHttp:
            def __init__(self):
                self.closed = False

            @staticmethod
            def start():
                raise COLLECTOR.CollectorError("synthetic HTTP start failure")

            def close(self):
                self.closed = True

        runtime = COLLECTOR.SystemRuntime(
            self.config, self.root, self.guard, Snapshots()
        )
        http = FailingHttp()
        journal = FakeJournal(BOOT_ID)
        runtime.http = http
        with (
            mock.patch.object(runtime, "_validate_docker_identity"),
            mock.patch.object(COLLECTOR, "JournalSource", return_value=journal),
        ):
            with self.assertRaisesRegex(
                COLLECTOR.CollectorError, "synthetic HTTP start failure"
            ):
                runtime.start()
        self.assertTrue(journal.started)
        self.assertTrue(journal.closed)
        self.assertTrue(http.closed)
        self.assertIsNone(runtime.journal)
        self.assertEqual(runtime._boot_id, "")

    def test_system_runtime_close_releases_journal_after_http_close_error(self):
        class Snapshots:
            client_path = Path("/tmp/sq8-client.py")
            credential_path = Path("/tmp/sq8-credential")

        class FailingHttp:
            @staticmethod
            def close():
                raise RuntimeError("synthetic HTTP close failure")

        class FakeJournal:
            closed = False

            def close(self):
                self.closed = True

        runtime = COLLECTOR.SystemRuntime(
            self.config, self.root, self.guard, Snapshots()
        )
        journal = FakeJournal()
        runtime.http = FailingHttp()
        runtime.journal = journal
        with self.assertRaisesRegex(RuntimeError, "synthetic HTTP close failure"):
            runtime.close()
        self.assertTrue(journal.closed)
        self.assertIsNone(runtime.journal)

    def test_journal_source_close_releases_reader_and_is_idempotent(self):
        reader = mock.Mock()
        source = COLLECTOR.JournalSource(BOOT_ID)
        source.reader = reader
        source.cursor = "cursor"
        source.close()
        reader.close.assert_called_once_with()
        self.assertIsNone(source.reader)
        self.assertIsNone(source.cursor)
        source.close()

    def test_config_rejects_noncanonical_negative_schedule(self):
        config = {
            "schema_version": COLLECTOR.CONFIG_SCHEMA,
            "run_id": "run",
            "identities": self.config.identities,
            "input_files": [
                {
                    "path": "fixtures/resource.json",
                    "source_file": str(self.config.input_files[0].source_file),
                }
            ],
            "phase_artifacts": {
                key: str(value) for key, value in self.config.phase_artifacts.items()
            },
            "http": {
                "target": "/v1/chat/completions",
                "resource_body_template": self.config.resource_body_template,
                "negative_cases": [
                    {
                        "after_request": 24,
                        "name": "context_overflow_1",
                        "body_base64": base64.b64encode(b"{}").decode(),
                        "expected_status": 400,
                    },
                    {
                        "after_request": 50,
                        "name": "malformed_json",
                        "body_base64": base64.b64encode(b"{").decode(),
                        "expected_status": 400,
                    },
                    {
                        "after_request": 75,
                        "name": "context_overflow_2",
                        "body_base64": base64.b64encode(b"{}").decode(),
                        "expected_status": 400,
                    },
                ],
            },
            "restart_command": ["unused"],
            "ready_url": COLLECTOR.HTTP_READY_URL,
            "amd_smi": COLLECTOR.AMD_SMI_BIN,
        }
        path = self._write("bad-config.json", compact(config))
        with self.assertRaisesRegex(
            COLLECTOR.CollectorError, "negative request schedule"
        ):
            COLLECTOR.load_config(path)

    def test_config_rejects_false_context_overflow_body(self):
        document = self._valid_config_document()
        document["http"]["negative_cases"][0]["body_base64"] = base64.b64encode(
            compact(
                {
                    "model": self.config.resource_body_template["model"],
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 2,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 0,
                }
            )
        ).decode("ascii")
        path = self._write("bad-overflow-config.json", compact(document))
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "context-overflow"):
            COLLECTOR.load_config(path)

    def test_config_requires_syntactically_malformed_json_negative(self):
        document = self._valid_config_document()
        document["http"]["negative_cases"][1]["body_base64"] = base64.b64encode(
            b'{"duplicate":1,"duplicate":2}'
        ).decode("ascii")
        path = self._write("duplicate-key-negative-config.json", compact(document))
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "must be malformed JSON"):
            COLLECTOR.load_config(path)

    def test_config_requires_every_real_phase_artifact_and_creates_nothing(self):
        missing = self.root / "not-produced" / "cancel-results.json"
        document = self._valid_config_document()
        document["phase_artifacts"]["cancel-results.json"] = str(missing)
        path = self._write("missing-artifact-config.json", compact(document))
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "phase artifact"):
            COLLECTOR.load_config(path)
        self.assertFalse(missing.exists())
        self.assertFalse(self.output.exists())

    def test_valid_config_binds_config_and_collector_implementation_inputs(self):
        path = self._write("valid-config.json", compact(self._valid_config_document()))
        loaded = COLLECTOR.load_config(path)
        input_paths = [item.path for item in loaded.input_files]
        self.assertEqual(
            input_paths, sorted(input_paths, key=lambda item: item.encode("utf-8"))
        )
        self.assertIn("collector/config.json", input_paths)
        self.assertIn(COLLECTOR.RESOURCE_FIXTURE_INPUT_PATH, input_paths)
        self.assertIn("tools/collect-sq8-openwebui-release.py", input_paths)
        self.assertIn(COLLECTOR.HTTP_CLIENT_INPUT_PATH, input_paths)
        config_input = next(
            item for item in loaded.input_files if item.path == "collector/config.json"
        )
        self.assertTrue(config_input.source_file.samefile(path))

    def _valid_config_document(self):
        return {
            "schema_version": COLLECTOR.CONFIG_SCHEMA,
            "run_id": "run",
            "identities": self.config.identities,
            "input_files": [
                {
                    "path": "fixtures/resource.json",
                    "source_file": str(self.config.input_files[0].source_file),
                }
            ],
            "phase_artifacts": {
                key: str(value) for key, value in self.config.phase_artifacts.items()
            },
            "http": {
                "target": "/v1/chat/completions",
                "resource_body_template": self.config.resource_body_template,
                "negative_cases": [
                    {
                        "after_request": item.after_request,
                        "name": item.name,
                        "body_base64": base64.b64encode(item.body).decode(),
                        "expected_status": item.expected_status,
                    }
                    for item in self.config.negative_cases
                ],
            },
            "restart_command": ["unused"],
            "ready_url": COLLECTOR.HTTP_READY_URL,
            "amd_smi": COLLECTOR.AMD_SMI_BIN,
        }

    def test_lifecycle_observer_validates_credentials_and_matches_journal_bytes(self):
        observer_dir = self.root / "observer"
        observer_dir.mkdir(mode=0o700)
        observer_path = observer_dir / "lifecycle-observer.sock"
        observer = COLLECTOR.LifecycleObserver(
            observer_path,
            self.guard,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )
        observer.open()
        try:
            event = {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "request_progress",
                "observed_monotonic_ns": time.monotonic_ns(),
                "request_id": "req-observer",
                "completion_id": "chatcmpl-observer",
                "phase": "prefill",
                "processed_prompt_tokens": 128,
                "prompt_tokens": 4096,
            }
            payload = json.dumps(
                event,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(payload, os.fspath(observer_path))
            datagram = observer.receive(
                time.monotonic_ns() + 1_000_000_000,
                expected_sender_pid=os.getpid(),
            )
            self.assertEqual(datagram.raw_payload, payload)
            self.assertGreaterEqual(datagram.mirror_delay_ns, 0)
            self.assertEqual(datagram.sender_uid, os.geteuid())
            correlator = COLLECTOR.ObserverJournalCorrelator()
            correlator.observe(datagram)
            correlated = correlator.correlate_journal_message(
                "INFO:     " + payload.decode("ascii"),
                event,
            )
            self.assertEqual(
                correlated.received_monotonic_ns, datagram.received_monotonic_ns
            )
            correlator.require_complete()
            observer.require_empty()
            self.assertEqual(observer_path.stat().st_mode & 0o777, 0o600)
        finally:
            observer.close()
        self.assertFalse(observer_path.exists())

    def test_lifecycle_observer_rejects_wrong_sender_pid(self):
        observer_dir = self.root / "observer-wrong-pid"
        observer_dir.mkdir(mode=0o700)
        observer_path = observer_dir / "lifecycle-observer.sock"
        observer = COLLECTOR.LifecycleObserver(
            observer_path,
            self.guard,
            expected_uid=os.geteuid(),
        )
        observer.open()
        try:
            event = {
                "schema_version": COLLECTOR.LIFECYCLE_SCHEMA,
                "event": "request_started",
                "observed_monotonic_ns": time.monotonic_ns(),
                "request_id": "req-observer",
                "completion_id": "chatcmpl-observer",
                "stream": True,
                "prompt_tokens": 32,
                "admit_to_start_ns": 1,
            }
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(compact(event), os.fspath(observer_path))
            with self.assertRaisesRegex(COLLECTOR.CollectorError, "sender PID or UID"):
                observer.receive(
                    time.monotonic_ns() + 1_000_000_000,
                    expected_sender_pid=os.getpid() + 1,
                )
        finally:
            observer.close()

    def test_stable_kfd_snapshot_cross_checks_worker_and_rejects_other_owner(self):
        kfd = self.root / "kfd"
        worker = kfd / "1201"
        idle = kfd / "1300"
        worker.mkdir(parents=True)
        idle.mkdir()
        (worker / "vram_51545").write_text("20000000000\n", encoding="ascii")
        (idle / "vram_51545").write_text("0\n", encoding="ascii")
        self.assertEqual(
            COLLECTOR.capture_stable_kfd_vram(kfd, 1201),
            {1201: 20_000_000_000, 1300: 0},
        )
        (idle / "vram_51545").write_text("4096\n", encoding="ascii")
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "unrelated process"):
            COLLECTOR.capture_stable_kfd_vram(kfd, 1201)

    def test_live_process_executable_hash_is_bound_by_starttime_and_inode(self):
        _, starttime = COLLECTOR.process_identity(Path("/proc"), os.getpid())
        observed = COLLECTOR.hash_live_process_executable(
            Path("/proc"), os.getpid(), starttime
        )
        digest = hashlib.sha256()
        with Path(f"/proc/{os.getpid()}/exe").open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        self.assertEqual(observed, digest.hexdigest())
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "starttime differs"):
            COLLECTOR.hash_live_process_executable(
                Path("/proc"), os.getpid(), starttime + 1
            )

    def test_control_group_rejects_traversal_and_reads_with_openat(self):
        root = self.root / "cgroup"
        group = root / "system.slice" / "ullm-openai.service"
        group.mkdir(parents=True)
        (group / "memory.current").write_text("123456\n", encoding="ascii")
        self.assertEqual(
            COLLECTOR.read_cgroup_memory_current(
                root, "/system.slice/ullm-openai.service"
            ),
            123456,
        )
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "safe absolute"):
            COLLECTOR.read_cgroup_memory_current(root, "/system.slice/../escape")

    def test_restart_hook_fields_reject_common_field_injection(self):
        fields = {
            "injection": "post_header_worker_kill",
            "target_pid": 1201,
            "target_starttime_ticks": 10001,
            "signal": "SIGKILL",
            "command": "kill -KILL -- 1201",
            "started_monotonic_ns": 10,
            "completed_monotonic_ns": 11,
            "sequence": 0,
        }
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "fields differ"):
            COLLECTOR.validate_hook_fields("fault_injection", fields)

    def test_http_adapter_accepts_strict_fake_command_stream(self):
        fake = self.root / "fake-http-client.py"
        fake.write_text(
            textwrap.dedent(
                """
                import base64
                import hashlib
                import json
                import sys

                EVENT = "ullm.sq8.openwebui_http_client.event.v1"
                def emit(event, **fields):
                    print(json.dumps({"schema_version": EVENT, "event": event, **fields}, separators=(",", ":")), flush=True)

                emit("ready", observed_monotonic_ns=1)
                for line in sys.stdin:
                    command = json.loads(line)
                    if command["command"] == "shutdown":
                        emit("shutdown_complete", observed_monotonic_ns=99)
                        raise SystemExit(0)
                    body = base64.b64decode(command["body_base64"])
                    emit(
                        "http_request",
                        request_key=command["request_key"],
                        method="POST",
                        target=command["target"],
                        headers={"content_type":"application/json","content_length":len(body),"authorization_mode":"valid_bearer"},
                        body_base64=command["body_base64"],
                        body_sha256=hashlib.sha256(body).hexdigest(),
                        body_bytes=len(body),
                        connect_completed_monotonic_ns=10,
                        write_started_monotonic_ns=11,
                        last_body_byte_sent_monotonic_ns=12,
                    )
                    emit(
                        "http_response_start",
                        request_key=command["request_key"],
                        status=200,
                        headers=[["Content-Type","text/event-stream"]],
                        observed_monotonic_ns=13,
                    )
                    response = (
                        b'data: {"id":"chatcmpl-fake","choices":[{"delta":{"content":"x"}}]}\\n\\n'
                        b'data: {"id":"chatcmpl-fake","choices":[],"usage":{"completion_tokens":2}}\\n\\n'
                        b'data: [DONE]\\n\\n'
                    )
                    emit(
                        "http_body_chunk",
                        request_key=command["request_key"],
                        chunk_index=0,
                        body_base64=base64.b64encode(response).decode(),
                        body_sha256=hashlib.sha256(response).hexdigest(),
                        body_bytes=len(response),
                        observed_monotonic_ns=14,
                    )
                    emit(
                        "http_response_end",
                        request_key=command["request_key"],
                        outcome="eof",
                        error=None,
                        body_bytes=len(response),
                        body_sha256=hashlib.sha256(response).hexdigest(),
                        observed_monotonic_ns=15,
                    )
                """
            ),
            encoding="utf-8",
        )
        process = COLLECTOR.HttpClientProcess((sys.executable, str(fake)), self.guard)
        body = compact(
            {
                "model": "Qwen3-14B-SQ8",
                "messages": [{"role": "user", "content": "fixture"}],
                "stream": True,
            }
        )
        plan = COLLECTOR.HttpPlan(
            "resource_normal",
            "fake-case",
            1,
            "fake-request",
            "/v1/chat/completions",
            body,
            200,
            True,
        )
        events = []
        process.start()
        try:
            observation = process.request(
                plan, lambda event, fields: events.append((event, fields))
            )
            self.assertEqual(observation.completion_id, "chatcmpl-fake")
            self.assertEqual(observation.outcome, "eof")
            self.assertEqual(
                [event for event, _ in events],
                [
                    "http_request",
                    "http_response_start",
                    "http_body_chunk",
                    "http_response_end",
                ],
            )
        finally:
            process.close()

    def test_http_adapter_runs_actual_client_and_shutdown_contract(self):
        received = {}
        response = (
            b'data: {"id":"chatcmpl-actual","choices":[{"delta":{"content":"x"}}]}\n\n'
            b'data: {"id":"chatcmpl-actual","choices":[],"usage":{"completion_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                received["path"] = self.path
                received["authorization"] = self.headers.get("Authorization")
                received["body"] = self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format, *_args):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        credential = self._write("actual-client-key", SECRET + b"\n")
        credential.chmod(0o600)
        process = COLLECTOR.HttpClientProcess(
            (
                sys.executable,
                str(ROOT / "tools" / "sq8-openwebui-http-client.py"),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--api-key-file",
                str(credential),
            ),
            self.guard,
        )
        body = compact(
            {
                "model": "Qwen3-14B-SQ8",
                "messages": [{"role": "user", "content": "fixture"}],
                "stream": True,
            }
        )
        plan = COLLECTOR.HttpPlan(
            "resource_normal",
            "actual-case",
            1,
            "actual-request",
            "/v1/chat/completions",
            body,
            200,
            True,
        )
        events = []
        process.start()
        try:
            observation = process.request(
                plan, lambda event, fields: events.append((event, fields))
            )
            self.assertEqual(observation.completion_id, "chatcmpl-actual")
            self.assertEqual(observation.outcome, "eof")
        finally:
            try:
                process.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(received["authorization"], "Bearer " + SECRET.decode())
        self.assertEqual(received["body"], body)
        self.assertEqual(
            [event for event, _ in events],
            [
                "http_request",
                "http_response_start",
                "http_body_chunk",
                "http_response_end",
            ],
        )

    def test_http_adapter_rejects_nonzero_exit_after_shutdown_ack(self):
        fake = self.root / "nonzero-shutdown-client.py"
        fake.write_text(
            textwrap.dedent(
                """
                import json
                import sys

                schema = "ullm.sq8.openwebui_http_client.event.v1"
                print(json.dumps({"schema_version": schema, "event": "ready", "observed_monotonic_ns": 1}), flush=True)
                json.loads(sys.stdin.readline())
                print(json.dumps({"schema_version": schema, "event": "shutdown_complete", "observed_monotonic_ns": 2}), flush=True)
                raise SystemExit(7)
                """
            ),
            encoding="utf-8",
        )
        process = COLLECTOR.HttpClientProcess((sys.executable, str(fake)), self.guard)
        process.start()
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "exited nonzero"):
            process.close()

    def test_http_adapter_rejects_exit_before_shutdown_handshake(self):
        fake = self.root / "early-exit-client.py"
        fake.write_text(
            textwrap.dedent(
                """
                import json

                schema = "ullm.sq8.openwebui_http_client.event.v1"
                print(json.dumps({"schema_version": schema, "event": "ready", "observed_monotonic_ns": 1}), flush=True)
                """
            ),
            encoding="utf-8",
        )
        process = COLLECTOR.HttpClientProcess((sys.executable, str(fake)), self.guard)
        process.start()
        assert process.process is not None
        process.process.wait(timeout=2)
        with self.assertRaisesRegex(
            COLLECTOR.CollectorError, "before the shutdown handshake"
        ):
            process.close()

    def test_http_adapter_rejects_extra_event_after_shutdown_ack(self):
        fake = self.root / "extra-shutdown-event-client.py"
        fake.write_text(
            textwrap.dedent(
                """
                import json
                import sys

                schema = "ullm.sq8.openwebui_http_client.event.v1"
                print(json.dumps({"schema_version": schema, "event": "ready", "observed_monotonic_ns": 1}), flush=True)
                json.loads(sys.stdin.readline())
                ack = json.dumps({"schema_version": schema, "event": "shutdown_complete", "observed_monotonic_ns": 2})
                print(ack)
                print(ack, flush=True)
                """
            ),
            encoding="utf-8",
        )
        process = COLLECTOR.HttpClientProcess((sys.executable, str(fake)), self.guard)
        process.start()
        with self.assertRaisesRegex(COLLECTOR.CollectorError, "emitted data after"):
            process.close()


if __name__ == "__main__":
    unittest.main()
