from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_openwebui_gate_ingest.py"
GATE_TEST_PATH = ROOT / "tests" / "test_run_openwebui_soak_gate.py"
GATE_SOURCE = ROOT / "tools" / "run-openwebui-soak-gate.py"
SUPPORT_SOURCE = ROOT / "tools" / "run-openwebui-stop-gate.py"
BROWSER_SCRIPT = ROOT / "deploy" / "openwebui" / "browser-soak.cjs"
DIRECT_TEST_PATH = ROOT / "tests" / "test_run_sq8_direct_cancel_gate.py"
DIRECT_GATE_SOURCE = ROOT / "tools" / "run-sq8-direct-cancel-gate.py"
COLLECTOR_SOURCE = ROOT / "tools" / "collect-sq8-openwebui-release.py"
HTTP_CLIENT_SOURCE = ROOT / "tools" / "sq8-openwebui-http-client.py"
DIRECT_PILOT = Path("/home/homelab1/datapool/sq8-direct-cancel-formal-20260711-194417")


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INGEST = load_path("sq8_openwebui_gate_ingest", MODULE_PATH)
GATE_TEST = load_path("_sq8_ingest_gate_fixture", GATE_TEST_PATH)
GATE = GATE_TEST.TOOL
DIRECT_TEST = load_path("_sq8_direct_ingest_fixture", DIRECT_TEST_PATH)
DIRECT_GATE = DIRECT_TEST.GATE
assert DIRECT_GATE is not None


def digest(raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_private(path, raw, mode=0o600):
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


class CombinedBundle:
    base_url = "http://127.0.0.1:3000"
    service = "ullm-openai.service"
    boot_id = "b" * 32
    pid = 4242
    image = "ghcr.io/ullm/browser@sha256:" + "a" * 64
    image_content_id = "sha256:" + "a" * 64

    def __init__(self, parent):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self.root = parent / "combined-gate"
        self.browser = self.root / "browser"
        self.root.mkdir(mode=0o700)
        self.browser.mkdir(mode=0o700)
        self.cases, browser_summary = GATE_TEST.browser_values(
            self.base_url, include_smoke=True
        )
        self.browser_lines = GATE_TEST.framed_lines(self.cases, browser_summary)
        browser_stdout = b"".join(raw + b"\n" for raw, _value in self.browser_lines)
        browser_summary_raw = self.browser_lines[-1][0] + b"\n"
        write_private(self.browser / "browser-stdout.jsonl", browser_stdout)
        write_private(
            self.browser / "openwebui-soak-summary.json", browser_summary_raw, 0o400
        )

        browser_evidence = []
        for index, (raw, value) in zip(
            GATE.case_indices(include_smoke=True),
            self.browser_lines[:-1],
            strict=True,
        ):
            browser_evidence.append(
                GATE.validate_browser_case(
                    value,
                    raw,
                    GATE.SecretGuard([]),
                    case_index=index,
                    base_url=self.base_url,
                    include_smoke=True,
                )
            )
        browser_result = {
            "chat_count": 21,
            "action_count": 105,
            "socket_event_count": 84,
            "browser_summary_bytes": len(browser_summary_raw),
            "browser_summary_sha256": digest(browser_summary_raw),
            "mode": GATE.COMBINED_MODE,
            "schedule": GATE.schedule_evidence(include_smoke=True),
        }

        machine = GATE.SoakLifecycleMachine(expected_count=21)
        journal_lines = []
        observer_lines = []
        sequence = 0
        for index in GATE.case_indices(include_smoke=True):
            request_id = f"request-secret-{index}"
            completion_id = f"completion-secret-{index}"
            for event in GATE_TEST.lifecycle_trace(
                request_id, completion_id, index * 1000 + 21
            ):
                machine.consume(event)
                payload = GATE.compact_json(event)
                observer_lines.append(payload)
                monotonic_usec = (event["observed_monotonic_ns"] + 999) // 1000
                journal_lines.append(
                    GATE.compact_json(
                        {
                            "__CURSOR": f"cursor-{sequence:03d}",
                            "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
                            "_BOOT_ID": self.boot_id,
                            "_PID": str(self.pid),
                            "_SYSTEMD_UNIT": self.service,
                            "PRIORITY": "6",
                            "MESSAGE": "INFO:     " + payload.decode("ascii"),
                            "_COMM": "ullm-openai",
                        }
                    )
                )
                sequence += 1
        observer_raw = b"".join(raw + b"\n" for raw in observer_lines)
        journal_raw = b"".join(raw + b"\n" for raw in journal_lines)
        write_private(self.root / "observer.raw.jsonl", observer_raw)
        write_private(self.root / "service-journal.raw.jsonl", journal_raw)

        correlations = GATE.validate_gateway_traces(
            machine, browser_evidence, include_smoke=True
        )
        empty_sha = digest(b"")
        self.summary = {
            "schema_version": GATE.COMBINED_GATE_SCHEMA,
            "passed": True,
            "service": {
                "unit_sha256": digest(self.service),
                "main_pid_sha256": digest(str(self.pid)),
                "user_uid_sha256": digest(str(os.getuid())),
                "user_gid_sha256": digest(str(os.getgid())),
                "boot_id_sha256": digest(self.boot_id),
                "restart_count": 7,
                "identity_invariant": True,
            },
            "browser": {
                "image_reference_sha256": digest(self.image),
                "image_content_digest": self.image_content_id,
                "script_sha256": digest(BROWSER_SCRIPT.read_bytes()),
                "gate_source_sha256": digest(GATE_SOURCE.read_bytes()),
                "support_source_sha256": digest(SUPPORT_SOURCE.read_bytes()),
                **browser_result,
                "stdout_lines": 22,
                "stdout_bytes": len(browser_stdout),
                "stdout_sha256": digest(browser_stdout),
                "stderr_bytes": 0,
                "stderr_sha256": empty_sha,
            },
            "gateway": {
                "request_count": 21,
                "maximum_active_requests": 1,
                "stop_release_count": 21,
                "reset_complete_count": 21,
                "every_admission_after_previous_release": True,
                "correlations": correlations,
            },
            "artifacts": {
                "observer": {
                    "file": "observer.raw.jsonl",
                    "bytes": len(observer_raw),
                    "records": len(observer_lines),
                    "sha256": digest(observer_raw),
                },
                "journal": {
                    "file": "service-journal.raw.jsonl",
                    "bytes": len(journal_raw),
                    "records": len(journal_lines),
                    "sha256": digest(journal_raw),
                    "unique_cursors": len(journal_lines),
                    "lifecycle_records": len(observer_lines),
                    "stderr_bytes": 0,
                    "stderr_sha256": empty_sha,
                },
                "browser_stdout": {
                    "file": "browser/browser-stdout.jsonl",
                    "bytes": len(browser_stdout),
                    "records": 22,
                    "sha256": digest(browser_stdout),
                },
                "browser_summary": {
                    "file": "browser/openwebui-soak-summary.json",
                    "bytes": len(browser_summary_raw),
                    "sha256": digest(browser_summary_raw),
                },
            },
            "mode": GATE.COMBINED_MODE,
            "schedule": GATE.schedule_evidence(include_smoke=True),
        }
        self.write_summary()
        self.root.chmod(0o700)
        self.browser.chmod(0o700)

    def write_summary(self):
        write_private(
            self.root / "summary.json", GATE.compact_json(self.summary) + b"\n"
        )

    def bindings(self, *, forbidden_values=()):
        return INGEST.GateInputBindings(
            gate_source=GATE_SOURCE,
            gate_source_sha256=digest(GATE_SOURCE.read_bytes()),
            support_source=SUPPORT_SOURCE,
            support_source_sha256=digest(SUPPORT_SOURCE.read_bytes()),
            browser_script=BROWSER_SCRIPT,
            browser_script_sha256=digest(BROWSER_SCRIPT.read_bytes()),
            browser_image_reference=self.image,
            browser_image_content_id=self.image_content_id,
            openwebui_base_url=self.base_url,
            service_unit=self.service,
            boot_id=self.boot_id,
            gateway_pid=self.pid,
            uid=os.getuid(),
            gid=os.getgid(),
            restart_count=7,
            forbidden_values=forbidden_values,
        )

    def browser_stdout_values(self):
        return [
            json.loads(raw)
            for raw in (self.browser / "browser-stdout.jsonl").read_bytes().splitlines()
        ]

    def write_browser_stdout_values(self, values):
        write_private(
            self.browser / "browser-stdout.jsonl",
            b"".join(
                json.dumps(value, separators=(",", ":")).encode() + b"\n"
                for value in values
            ),
        )

    def journal_values(self):
        return [
            json.loads(raw)
            for raw in (self.root / "service-journal.raw.jsonl")
            .read_bytes()
            .splitlines()
        ]

    def write_journal_values(self, values):
        write_private(
            self.root / "service-journal.raw.jsonl",
            b"".join(GATE.compact_json(value) + b"\n" for value in values),
        )


class DirectCancelBundle:
    secret = b"direct-cancel-api-secret-test-value"
    image_id = "sha256:" + "a" * 64
    network_id = "b" * 64
    service = "ullm-openai.service"
    service_user = "homelab1"
    boot_id = "c" * 32
    control_group = "/system.slice/ullm-openai.service"
    gateway_pid = 5200
    worker_pid = 5201

    def __init__(self, parent):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self.root = parent / "direct-cancel"
        self.root.mkdir(mode=0o700)
        fixture_root = ROOT / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
        self.fixtures = {
            fixture_id: DIRECT_GATE.load_fixture(
                fixture_root / f"{fixture_id}.json", fixture_id
            )
            for fixture_id in ("exact-p0032", "exact-p3584")
        }
        self.http_values = [
            {
                "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
                "event": "ready",
                "observed_monotonic_ns": 1,
            }
        ]
        self.observer_values = []
        self.journal_values = []
        self.correlation_values = []
        self.request_summaries = []
        sequence = 0
        for phase in DIRECT_GATE.PHASE_ORDER:
            for role in ("target", "recovery"):
                request_index = len(self.request_summaries) + 1
                base = request_index * 1_000_000
                request_key = f"direct-{phase}-{role}"
                request_id = f"direct-request-secret-{request_index}"
                completion_id = f"chatcmpl-direct-secret-{request_index}"
                if role == "target":
                    spec = DIRECT_GATE.PHASE_SPECS[phase]
                    fixture = self.fixtures[spec.fixture_id]
                    max_tokens = 512
                    events, request_summary, content_ns = self._target_trace(
                        phase, base, request_id, completion_id
                    )
                else:
                    fixture = self.fixtures["exact-p0032"]
                    max_tokens = 2
                    events, request_summary = self._recovery_trace(
                        phase, base, request_id, completion_id
                    )
                    content_ns = None
                body = DIRECT_GATE.request_body(fixture, max_tokens)
                self.http_values.extend(
                    self._http_request_values(
                        request_key,
                        body,
                        base,
                        role,
                        completion_id,
                        content_ns,
                    )
                )
                self.request_summaries.append(request_summary)
                for event in events:
                    payload = DIRECT_GATE.compact_json(event)
                    self.observer_values.append(event)
                    monotonic_usec = (event["observed_monotonic_ns"] + 999) // 1000
                    journal = {
                        "__CURSOR": f"direct-cursor-{sequence:03d}",
                        "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
                        "_BOOT_ID": self.boot_id,
                        "_PID": str(self.gateway_pid),
                        "_SYSTEMD_UNIT": self.service,
                        "PRIORITY": "6",
                        "MESSAGE": payload.decode("ascii"),
                    }
                    self.journal_values.append(journal)
                    self.correlation_values.append(
                        {
                            "schema_version": DIRECT_GATE.GATE_SCHEMA,
                            "sequence": sequence,
                            "cursor": journal["__CURSOR"],
                            "journal_monotonic_usec": journal["__MONOTONIC_TIMESTAMP"],
                            "journal_pid": journal["_PID"],
                            "observer_received_monotonic_ns": event[
                                "observed_monotonic_ns"
                            ]
                            + 10,
                            "observer_sender_pid": self.gateway_pid,
                            "observer_sender_uid": os.getuid(),
                            "observer_sender_gid": os.getgid(),
                            "payload_sha256": digest(payload),
                            "payload_bytes": len(payload),
                        }
                    )
                    sequence += 1
        self.http_values.append(
            {
                "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
                "event": "shutdown_complete",
                "observed_monotonic_ns": 9_000_000,
            }
        )
        self._write_jsonl("http-client.raw.jsonl", self.http_values)
        self._write_jsonl("observer.raw.jsonl", self.observer_values)
        self._write_jsonl("service-journal.raw.jsonl", self.journal_values)
        self._write_jsonl(
            "observer-journal-correlation.raw.jsonl", self.correlation_values
        )
        self.manifest = self._manifest()
        self._write_document("input-manifest.json", self.manifest)
        self.summary = self._summary()
        self._write_document("summary.json", self.summary)

    def _target_trace(self, phase, base, request_id, completion_id):
        prompt_tokens = DIRECT_TEST.TARGET_PROMPT_TOKENS[phase]
        admitted_at = base + 100
        events = [
            DIRECT_TEST.admitted(
                admitted_at,
                prompt_tokens,
                512,
                request_id=request_id,
                completion_id=completion_id,
            ),
            DIRECT_TEST.started(
                admitted_at + 1,
                prompt_tokens,
                request_id=request_id,
                completion_id=completion_id,
            ),
        ]
        progress_values = []
        if phase == "prefill_after_128":
            progress_values = [128]
        elif phase == "prefill_after_2048":
            progress_values = list(range(128, 2049, 128))
        elif phase == "decode_after_first_content":
            progress_values = [32]
        for offset, processed in enumerate(progress_values, start=2):
            events.append(
                DIRECT_TEST.progress(
                    admitted_at + offset,
                    processed,
                    prompt_tokens,
                    request_id=request_id,
                    completion_id=completion_id,
                )
            )
        if phase == "decode_after_first_content":
            events.append(
                DIRECT_TEST.first_token(
                    admitted_at + 3,
                    request_id=request_id,
                    completion_id=completion_id,
                )
            )
            trigger_ns = admitted_at + 4
            completion_tokens = 1
        elif progress_values:
            trigger_ns = events[-1]["observed_monotonic_ns"]
            completion_tokens = 0
        else:
            trigger_ns = admitted_at + 1
            completion_tokens = 0
        cancel_ns = trigger_ns + 1
        release_ns = cancel_ns + 1
        events.extend(
            (
                DIRECT_TEST.cancel_requested(
                    cancel_ns,
                    admitted_at=admitted_at,
                    request_id=request_id,
                    completion_id=completion_id,
                ),
                DIRECT_TEST.released(
                    release_ns,
                    prompt_tokens,
                    cancelled=True,
                    completion_tokens=completion_tokens,
                    admitted_at=admitted_at,
                    request_id=request_id,
                    completion_id=completion_id,
                ),
            )
        )
        summary = {
            "phase": phase,
            "role": "target",
            "request_id": request_id,
            "completion_id": completion_id,
            "trigger_observed_monotonic_ns": trigger_ns,
            "client_close_monotonic_ns": trigger_ns,
            "cancel_observed_monotonic_ns": cancel_ns,
            "release_observed_monotonic_ns": release_ns,
            "progress": progress_values,
            "completion_tokens": completion_tokens,
        }
        return (
            events,
            summary,
            trigger_ns if phase == "decode_after_first_content" else None,
        )

    def _recovery_trace(self, phase, base, request_id, completion_id):
        admitted_at = base + 100
        events = [
            DIRECT_TEST.admitted(
                admitted_at,
                32,
                2,
                request_id=request_id,
                completion_id=completion_id,
            ),
            DIRECT_TEST.started(
                admitted_at + 1,
                32,
                request_id=request_id,
                completion_id=completion_id,
            ),
            DIRECT_TEST.progress(
                admitted_at + 2,
                32,
                32,
                request_id=request_id,
                completion_id=completion_id,
            ),
            DIRECT_TEST.first_token(
                admitted_at + 3,
                request_id=request_id,
                completion_id=completion_id,
            ),
            DIRECT_TEST.released(
                admitted_at + 4,
                32,
                cancelled=False,
                completion_tokens=2,
                admitted_at=admitted_at,
                request_id=request_id,
                completion_id=completion_id,
            ),
        ]
        return events, {
            "phase": phase,
            "role": "recovery",
            "request_id": request_id,
            "completion_id": completion_id,
            "release_observed_monotonic_ns": admitted_at + 4,
        }

    def _http_request_values(
        self, request_key, body, base, role, completion_id, content_ns
    ):
        request = {
            "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
            "event": "http_request",
            "request_key": request_key,
            "method": "POST",
            "target": DIRECT_GATE.HTTP_TARGET,
            "headers": {
                "content_type": "application/json",
                "content_length": len(body),
                "authorization_mode": "valid_bearer",
            },
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_sha256": digest(body),
            "body_bytes": len(body),
            "connect_completed_monotonic_ns": base + 1,
            "write_started_monotonic_ns": base + 2,
            "last_body_byte_sent_monotonic_ns": base + 3,
        }
        start = {
            "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_start",
            "request_key": request_key,
            "status": 200,
            "headers": [["Content-Type", "text/event-stream"]],
            "observed_monotonic_ns": base + 4,
        }
        if role == "target" and content_ns is None:
            response = b": hold\n\n"
            chunk_ns = base + 5
        elif role == "target":
            response = self._sse(
                {
                    "id": completion_id,
                    "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
                }
            )
            chunk_ns = content_ns
        else:
            response = b"".join(
                (
                    self._sse(
                        {
                            "id": completion_id,
                            "choices": [
                                {
                                    "delta": {"content": "ok"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ),
                    self._sse(
                        {
                            "id": completion_id,
                            "choices": [{"delta": {}, "finish_reason": "length"}],
                        }
                    ),
                    self._sse(
                        {
                            "id": completion_id,
                            "choices": [],
                            "usage": {"completion_tokens": 2},
                        }
                    ),
                    b"data: [DONE]\n\n",
                )
            )
            chunk_ns = base + 200
        chunk = {
            "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
            "event": "http_body_chunk",
            "request_key": request_key,
            "chunk_index": 0,
            "body_base64": base64.b64encode(response).decode("ascii"),
            "body_sha256": digest(response),
            "body_bytes": len(response),
            "observed_monotonic_ns": chunk_ns,
        }
        end = {
            "schema_version": DIRECT_GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_end",
            "request_key": request_key,
            "outcome": "client_closed" if role == "target" else "eof",
            "error": None,
            "body_bytes": len(response),
            "body_sha256": digest(response),
            "observed_monotonic_ns": base + 500,
        }
        return [request, start, chunk, end]

    @staticmethod
    def _sse(value):
        return b"data: " + DIRECT_GATE.compact_json(value) + b"\n\n"

    def _manifest(self):
        source_values = (
            ("tools/run-sq8-direct-cancel-gate.py", DIRECT_GATE_SOURCE),
            ("tools/collect-sq8-openwebui-release.py", COLLECTOR_SOURCE),
            ("tools/sq8-openwebui-http-client.py", HTTP_CLIENT_SOURCE),
        )
        fixture_values = [
            (
                "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/"
                f"{fixture_id}.json",
                self.fixtures[fixture_id].raw,
            )
            for fixture_id in ("exact-p0032", "exact-p3584")
        ]
        return {
            "schema_version": DIRECT_GATE.GATE_SCHEMA,
            "record_type": "input_manifest",
            "inputs": [
                {
                    "path": path,
                    "bytes": len(raw),
                    "sha256": digest(raw),
                }
                for path, raw in [
                    *((path, source.read_bytes()) for path, source in source_values),
                    *fixture_values,
                ]
            ],
            "request_bodies": [
                {
                    "fixture_id": fixture_id,
                    "max_tokens": max_tokens,
                    "bytes": len(
                        DIRECT_GATE.request_body(self.fixtures[fixture_id], max_tokens)
                    ),
                    "sha256": digest(
                        DIRECT_GATE.request_body(self.fixtures[fixture_id], max_tokens)
                    ),
                }
                for fixture_id, max_tokens in (
                    ("exact-p3584", 512),
                    ("exact-p0032", 512),
                    ("exact-p0032", 2),
                )
            ],
        }

    def _summary(self):
        artifact_names = (
            "http-client.raw.jsonl",
            "observer.raw.jsonl",
            "service-journal.raw.jsonl",
            "observer-journal-correlation.raw.jsonl",
        )
        return {
            "schema_version": DIRECT_GATE.GATE_SCHEMA,
            "record_type": "summary",
            "phase_order": list(DIRECT_GATE.PHASE_ORDER),
            "request_count": 8,
            "max_active": 1,
            "service_identity": {
                "unit": self.service,
                "user": self.service_user,
                "uid": os.getuid(),
                "gid": os.getgid(),
                "control_group": self.control_group,
                "gateway_pid": self.gateway_pid,
                "gateway_starttime_ticks": 100_000,
                "worker_pid": self.worker_pid,
                "worker_starttime_ticks": 100_001,
                "n_restarts": 2,
                "boot_id": self.boot_id,
            },
            "http_image_id": self.image_id,
            "docker_network_name": DIRECT_GATE.HTTP_NETWORK_NAME,
            "docker_network_id": self.network_id,
            "observer_socket": os.fspath(DIRECT_GATE.OBSERVER_SOCKET),
            "observer_event_count": 55,
            "journal_correlation_count": 55,
            "requests": copy.deepcopy(self.request_summaries),
            "artifacts": {
                name: {
                    "bytes": len((self.root / name).read_bytes()),
                    "lines": len((self.root / name).read_bytes().splitlines()),
                    "sha256": digest((self.root / name).read_bytes()),
                }
                for name in artifact_names
            },
        }

    def bindings(self, *, forbidden_values=None):
        return INGEST.DirectCancelInputBindings(
            gate_source=DIRECT_GATE_SOURCE,
            gate_source_sha256=digest(DIRECT_GATE_SOURCE.read_bytes()),
            collector_source=COLLECTOR_SOURCE,
            collector_source_sha256=digest(COLLECTOR_SOURCE.read_bytes()),
            http_client_source=HTTP_CLIENT_SOURCE,
            http_client_source_sha256=digest(HTTP_CLIENT_SOURCE.read_bytes()),
            http_image_id=self.image_id,
            docker_network_id=self.network_id,
            service_unit=self.service,
            service_user=self.service_user,
            boot_id=self.boot_id,
            control_group=self.control_group,
            gateway_pid=self.gateway_pid,
            gateway_starttime_ticks=100_000,
            worker_pid=self.worker_pid,
            worker_starttime_ticks=100_001,
            restart_count=2,
            uid=os.getuid(),
            gid=os.getgid(),
            forbidden_values=(self.secret,)
            if forbidden_values is None
            else forbidden_values,
        )

    def _write_jsonl(self, name, values):
        write_private(
            self.root / name,
            b"".join(DIRECT_GATE.compact_json(value) + b"\n" for value in values),
        )

    def _write_document(self, name, value):
        write_private(self.root / name, DIRECT_GATE.compact_json(value) + b"\n")

    def rewrite_summary(self):
        self._write_document("summary.json", self.summary)

    def rewrite_manifest(self):
        self._write_document("input-manifest.json", self.manifest)


class CombinedIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = CombinedBundle(self.temporary.name)

    def ingest(self, bindings=None):
        return INGEST.ingest_combined_soak_bundle(
            self.fixture.root,
            self.fixture.bindings() if bindings is None else bindings,
        )

    def test_exact_combined_bundle_converts_to_campaign_material(self):
        actions, claims, view = self.ingest()
        self.assertEqual(
            (self.fixture.browser / "openwebui-soak-summary.json").stat().st_mode
            & 0o777,
            0o400,
        )
        self.assertEqual(len(actions), 105)
        self.assertEqual(len(claims), 105)
        self.assertEqual(actions[0]["case_id"], "openwebui_smoke")
        self.assertEqual(actions[5]["case_id"], "openwebui_soak_chat_01")
        self.assertEqual(actions[-1]["case_id"], "openwebui_soak_chat_20")
        self.assertTrue(
            all(record["record_type"] == "browser_action" for record in actions)
        )
        self.assertTrue(all(record["phase"] == "openwebui" for record in actions))
        self.assertTrue(
            all(
                type(record["fields"]["started_monotonic_ns"]) is int
                and type(record["fields"]["completed_monotonic_ns"]) is int
                for record in actions
            )
        )
        self.assertEqual(claims[0].phase, "openwebui")
        self.assertEqual(claims[0].case_id, "openwebui_smoke")
        self.assertEqual(claims[5].case_id, "openwebui_soak_chat_01")
        self.assertEqual(claims[-1].case_id, "openwebui_soak_chat_20")
        self.assertTrue(claims[0].raw.startswith(b"{"))
        self.assertNotIn(b"\n", claims[0].raw)
        self.assertNotIn(b"_COMM", claims[0].raw)
        self.assertEqual(
            tuple(json.loads(claims[0].raw)), INGEST.REQUIRED_JOURNAL_FIELDS
        )
        self.assertEqual(view["mode"], "smoke_then_soak20")
        self.assertEqual(view["chat_count"], 21)
        self.assertEqual(view["action_count"], 105)
        self.assertEqual(len(view["cases"]), 21)
        self.assertIsInstance(view["cases"][0]["admitted_monotonic_ns"], int)
        encoded = GATE.compact_json(view)
        self.assertNotIn(b"request-secret-", encoded)
        self.assertNotIn(b"completion-secret-", encoded)
        self.assertNotIn(self.fixture.base_url.encode(), encoded)

    def test_converted_actions_are_globally_monotonic_and_tampering_fails(self):
        actions, _claims, _view = self.ingest()
        prior_completed = -1
        for position, record in enumerate(actions):
            fields = record["fields"]
            self.assertEqual(fields["action_index"], position % 5)
            self.assertGreaterEqual(fields["started_monotonic_ns"], prior_completed)
            self.assertGreaterEqual(
                fields["completed_monotonic_ns"], fields["started_monotonic_ns"]
            )
            prior_completed = fields["completed_monotonic_ns"]

        values = self.fixture.browser_stdout_values()
        values[1]["browser_actions"][0]["started_monotonic_ns"] = "1"
        values[1]["browser_actions"][0]["completed_monotonic_ns"] = "2"
        self.fixture.write_browser_stdout_values(values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_lifecycle_claims_are_the_campaign_contract_type(self):
        _actions, claims, _view = self.ingest()
        campaign = sys.modules["sq8_openwebui_campaign"]
        self.assertTrue(
            all(isinstance(claim, campaign.BundleLifecycleClaim) for claim in claims)
        )

        class StaticJournalSource:
            def __init__(self, rows):
                self.rows = iter(rows)

            def open_after(self, unit, boot_id):
                self.opened = (unit, boot_id)
                return "campaign-anchor"

            def read_next(self, timeout_usec):
                try:
                    return next(self.rows)
                except StopIteration:
                    time.sleep(min(timeout_usec / 1_000_000, 0.001))
                    return None

            def close(self):
                return None

        final_path = Path(self.temporary.name) / "campaign-journal.raw.jsonl"
        capture = campaign.CampaignJournalCapture(
            final_path,
            self.fixture.boot_id,
            campaign.PidEpoch(self.fixture.pid, self.fixture.pid + 1),
            scan_raw=lambda _raw, _label: None,
            source=StaticJournalSource([claim.raw for claim in claims]),
        )
        self.addCleanup(capture.abort)
        capture.start()
        claimed = capture.claim_bundle_records(
            claims, time.monotonic_ns() + 2_000_000_000
        )
        self.assertEqual(len(claimed), 105)
        hook = claimed[0].session_hook_record()
        self.assertEqual(hook["record_type"], "gateway_event")
        self.assertEqual(hook["fields"]["journal_pid"], self.fixture.pid)
        self.assertEqual(
            hook["fields"]["message_sha256"], digest(hook["fields"]["message"])
        )
        self.assertEqual(claimed[-1].case_id, "openwebui_soak_chat_20")

    def test_unknown_duplicate_and_reordered_cases_are_rejected(self):
        mutations = []
        unknown = self.fixture.browser_stdout_values()
        unknown[0]["browser_case"] = "unknown_case"
        mutations.append(unknown)
        duplicate = self.fixture.browser_stdout_values()
        duplicate[1] = copy.deepcopy(duplicate[0])
        mutations.append(duplicate)
        reordered = self.fixture.browser_stdout_values()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        mutations.append(reordered)
        for mutation_index, values in enumerate(mutations):
            with self.subTest(case=values[0]["browser_case"]):
                self.fixture.write_browser_stdout_values(values)
                with self.assertRaises(INGEST.GateIngestError):
                    self.ingest()
                self.fixture = CombinedBundle(
                    Path(self.temporary.name) / f"case-{mutation_index}"
                )

    def test_combined_schema_mode_and_schedule_are_fail_closed(self):
        for field, replacement in (
            ("schema_version", GATE.GATE_SCHEMA),
            ("mode", "soak20"),
            ("schedule", list(reversed(self.fixture.summary["schedule"]))),
        ):
            with self.subTest(field=field):
                original = copy.deepcopy(self.fixture.summary[field])
                self.fixture.summary[field] = replacement
                self.fixture.write_summary()
                with self.assertRaises(INGEST.GateIngestError):
                    self.ingest()
                self.fixture.summary[field] = original
                self.fixture.write_summary()

        values = self.fixture.browser_stdout_values()
        values[-1]["schedule"] = list(reversed(values[-1]["schedule"]))
        self.fixture.write_browser_stdout_values(values)
        write_private(
            self.fixture.browser / "openwebui-soak-summary.json",
            json.dumps(values[-1], separators=(",", ":")).encode() + b"\n",
            0o400,
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_summary_source_image_and_browser_file_bindings_are_rejected(self):
        mutations = (
            ("gate_source_sha256", "0" * 64),
            ("support_source_sha256", "1" * 64),
            ("script_sha256", "2" * 64),
            ("image_content_digest", "sha256:" + "3" * 64),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                original = self.fixture.summary["browser"][field]
                self.fixture.summary["browser"][field] = replacement
                self.fixture.write_summary()
                with self.assertRaises(INGEST.GateIngestError):
                    self.ingest()
                self.fixture.summary["browser"][field] = original
                self.fixture.write_summary()
        write_private(
            self.fixture.browser / "openwebui-soak-summary.json",
            b'{"changed":true}\n',
            0o400,
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "manifest-path")
        self.fixture.summary["artifacts"]["observer"]["file"] = "../observer.raw.jsonl"
        self.fixture.write_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_unmaterialized_browser_and_journal_stderr_must_be_empty(self):
        mutations = (
            self.fixture.summary["browser"],
            self.fixture.summary["artifacts"]["journal"],
        )
        for index, target in enumerate(mutations):
            with self.subTest(index=index):
                target["stderr_bytes"] = 1
                target["stderr_sha256"] = digest(b"diagnostic")
                self.fixture.write_summary()
                with self.assertRaisesRegex(INGEST.GateIngestError, "must be empty"):
                    self.ingest()
                target["stderr_bytes"] = 0
                target["stderr_sha256"] = digest(b"")
                self.fixture.write_summary()

    def test_journal_cursor_message_hash_and_pid_are_rejected(self):
        values = self.fixture.journal_values()
        values[1]["__CURSOR"] = values[0]["__CURSOR"]
        self.fixture.write_journal_values(values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "message")
        values = self.fixture.journal_values()
        values[0]["MESSAGE"] = "INFO: unrelated"
        self.fixture.write_journal_values(values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "pid")
        values = self.fixture.journal_values()
        values[0]["_PID"] = str(self.fixture.pid + 1)
        self.fixture.write_journal_values(values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "pid-type")
        values = self.fixture.journal_values()
        values[0]["_PID"] = self.fixture.pid
        self.fixture.write_journal_values(values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "hash")
        self.fixture.summary["artifacts"]["journal"]["sha256"] = "f" * 64
        self.fixture.write_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_file_replacement_after_read_is_rejected_at_seal(self):
        snapshot = INGEST.BundleSnapshot(
            self.fixture.root, uid=os.getuid(), gid=os.getgid()
        )
        self.addCleanup(snapshot.close)
        snapshot.read_small("summary", INGEST.MAX_SUMMARY_BYTES)
        snapshot.read_small("browser_summary", INGEST.MAX_SUMMARY_BYTES)
        list(snapshot.iter_lines("browser_stdout"))
        list(snapshot.iter_lines("journal"))
        list(snapshot.iter_lines("observer"))
        target = self.fixture.browser / "browser-stdout.jsonl"
        replacement = self.fixture.browser / "replacement"
        write_private(replacement, target.read_bytes())
        os.replace(replacement, target)
        with self.assertRaises(INGEST.GateIngestError):
            snapshot.seal()

    def test_bound_source_replacement_after_snapshot_is_rejected_at_seal(self):
        source = Path(self.temporary.name) / "bound-source.py"
        raw = b"print('bound source')\n"
        source.write_bytes(raw)
        snapshot = INGEST._StableSource(source, "test source", 1024, digest(raw))
        self.addCleanup(snapshot.close)
        replacement = source.with_name("replacement.py")
        replacement.write_bytes(raw)
        os.replace(replacement, source)
        with self.assertRaises(INGEST.GateIngestError):
            snapshot.seal()

    def test_forbidden_cleartext_is_rejected_across_streaming_chunks(self):
        secret = b"forbidden-secret-value"
        path = self.fixture.root / "summary.json"
        raw = path.read_bytes()
        padding = b" " * (INGEST.COPY_CHUNK_BYTES - 2 - len(raw))
        write_private(path, raw + padding + secret)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest(self.fixture.bindings(forbidden_values=(secret,)))

    def test_extra_layout_mode_and_hardlink_are_rejected(self):
        extra = self.fixture.root / "unexpected"
        write_private(extra, b"unexpected\n")
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()
        extra.unlink()

        self.fixture = CombinedBundle(Path(self.temporary.name) / "summary-mode")
        (self.fixture.browser / "openwebui-soak-summary.json").chmod(0o600)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        (self.fixture.root / "summary.json").chmod(0o640)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()
        (self.fixture.root / "summary.json").chmod(0o600)

        os.link(
            self.fixture.root / "summary.json",
            self.fixture.root.parent / "summary-hardlink",
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_expected_bundle_file_symlink_is_rejected(self):
        summary = self.fixture.root / "summary.json"
        outside = self.fixture.root.parent / "outside-summary.json"
        summary.rename(outside)
        summary.symlink_to(outside)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_input_source_hash_binding_is_rejected_before_bundle_conversion(self):
        bindings = dataclass_replace(
            self.fixture.bindings(), gate_source_sha256="0" * 64
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest(bindings)


class DirectCancelPublicContractTests(unittest.TestCase):
    def test_direct_cancel_ingest_contract_is_exposed(self):
        self.assertTrue(hasattr(INGEST, "DirectCancelInputBindings"))
        self.assertTrue(hasattr(INGEST, "DirectCancelIngestResult"))
        self.assertTrue(hasattr(INGEST, "ingest_direct_cancel_bundle"))


class DirectCancelIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = DirectCancelBundle(self.temporary.name)

    def ingest(self, bindings=None):
        return INGEST.ingest_direct_cancel_bundle(
            self.fixture.root,
            self.fixture.bindings() if bindings is None else bindings,
        )

    def fresh(self, name):
        self.fixture = DirectCancelBundle(Path(self.temporary.name) / name)
        return self.fixture

    def test_exact_direct_bundle_converts_http_claims_and_redacted_view(self):
        http_records, claims, view = self.ingest()
        self.assertEqual(len(http_records), 32)
        requests = [
            record for record in http_records if record["record_type"] == "http_request"
        ]
        self.assertEqual(
            [record["fields"]["request_index"] for record in requests],
            list(range(1, 9)),
        )
        self.assertEqual(
            [record["case_id"] for record in requests],
            [
                f"direct-{phase}-{role}"
                for phase in DIRECT_GATE.PHASE_ORDER
                for role in ("target", "recovery")
            ],
        )
        self.assertTrue(
            all(record["phase"] == "cancellation" for record in http_records)
        )
        self.assertEqual(len(claims), 55)
        expected_claim_counts = [4, 5, 5, 5, 20, 5, 6, 5]
        offset = 0
        for request, count in zip(requests, expected_claim_counts, strict=True):
            self.assertEqual(
                {claim.case_id for claim in claims[offset : offset + count]},
                {request["case_id"]},
            )
            offset += count
        self.assertTrue(all(claim.phase == "cancellation" for claim in claims))
        self.assertTrue(all(b"\n" not in claim.raw for claim in claims))
        self.assertEqual(view["request_count"], 8)
        self.assertEqual(view["lifecycle_record_count"], 55)
        self.assertEqual(len(view["cases"]), 8)
        encoded = DIRECT_GATE.compact_json(view)
        for forbidden in (
            self.fixture.secret,
            b"direct-request-secret-",
            b"chatcmpl-direct-secret-",
            DIRECT_GATE.request_body(self.fixture.fixtures["exact-p3584"], 512),
            base64.b64decode(self.fixture.http_values[3]["body_base64"], validate=True),
            os.fspath(DIRECT_GATE_SOURCE).encode(),
            self.fixture.boot_id.encode(),
            self.fixture.control_group.encode(),
        ):
            self.assertNotIn(forbidden, encoded)

    def test_direct_claims_are_consumed_by_campaign_capture(self):
        _http, claims, _view = self.ingest()
        campaign = sys.modules["sq8_openwebui_campaign"]

        class StaticJournalSource:
            def __init__(self, rows):
                self.rows = iter(rows)

            def open_after(self, _unit, _boot_id):
                return "direct-campaign-anchor"

            def read_next(self, timeout_usec):
                try:
                    return next(self.rows)
                except StopIteration:
                    time.sleep(min(timeout_usec / 1_000_000, 0.001))
                    return None

            def close(self):
                return None

        capture = campaign.CampaignJournalCapture(
            Path(self.temporary.name) / "direct-campaign-journal.raw.jsonl",
            self.fixture.boot_id,
            campaign.PidEpoch(self.fixture.gateway_pid, self.fixture.worker_pid),
            scan_raw=lambda _raw, _label: None,
            source=StaticJournalSource([claim.raw for claim in claims]),
        )
        self.addCleanup(capture.abort)
        capture.start()
        claimed = capture.claim_bundle_records(
            claims, time.monotonic_ns() + 2_000_000_000
        )
        self.assertEqual(len(claimed), 55)
        self.assertEqual(claimed[0].phase, "cancellation")
        self.assertEqual(
            claimed[0].case_id, "direct-after_started_before_progress-target"
        )
        self.assertEqual(
            claimed[-1].case_id, "direct-decode_after_first_content-recovery"
        )

    def test_exact_http_schedule_and_request_body_are_fail_closed(self):
        self.fixture.http_values[1]["request_key"] = "direct-unknown-target"
        self.fixture._write_jsonl("http-client.raw.jsonl", self.fixture.http_values)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("body")
        request = self.fixture.http_values[1]
        changed = DIRECT_GATE.request_body(self.fixture.fixtures["exact-p0032"], 512)
        request["body_base64"] = base64.b64encode(changed).decode("ascii")
        request["body_bytes"] = len(changed)
        request["body_sha256"] = digest(changed)
        request["headers"]["content_length"] = len(changed)
        self.fixture._write_jsonl("http-client.raw.jsonl", self.fixture.http_values)
        self.fixture.summary = self.fixture._summary()
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_lifecycle_semantic_and_case_mapping_mutation_is_rejected(self):
        mutation_index = 2
        event = self.fixture.observer_values[mutation_index]
        self.assertEqual(event["event"], "request_cancel_requested")
        event["reason"] = "server_shutdown"
        payload = DIRECT_GATE.compact_json(event)
        self.fixture.journal_values[mutation_index]["MESSAGE"] = payload.decode("ascii")
        correlation = self.fixture.correlation_values[mutation_index]
        correlation["payload_bytes"] = len(payload)
        correlation["payload_sha256"] = digest(payload)
        self.fixture._write_jsonl("observer.raw.jsonl", self.fixture.observer_values)
        self.fixture._write_jsonl(
            "service-journal.raw.jsonl", self.fixture.journal_values
        )
        self.fixture._write_jsonl(
            "observer-journal-correlation.raw.jsonl",
            self.fixture.correlation_values,
        )
        self.fixture.summary = self.fixture._summary()
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_manifest_summary_and_source_bindings_are_fail_closed(self):
        self.fixture.manifest["inputs"][0]["sha256"] = "0" * 64
        self.fixture.rewrite_manifest()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("summary")
        self.fixture.summary["phase_order"] = list(
            reversed(self.fixture.summary["phase_order"])
        )
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("source")
        bindings = dataclass_replace(
            self.fixture.bindings(), collector_source_sha256="f" * 64
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest(bindings)

        self.fresh("close-boundary")
        self.fixture.summary["requests"][0]["client_close_monotonic_ns"] = 0
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_correlation_payload_cursor_and_sender_mutations_are_rejected(self):
        mutations = (
            ("payload_sha256", "0" * 64),
            ("cursor", "other-cursor"),
            ("observer_sender_pid", self.fixture.gateway_pid + 1),
        )
        for index, (field, replacement) in enumerate(mutations):
            with self.subTest(field=field):
                self.fixture.correlation_values[0][field] = replacement
                self.fixture._write_jsonl(
                    "observer-journal-correlation.raw.jsonl",
                    self.fixture.correlation_values,
                )
                with self.assertRaises(INGEST.GateIngestError):
                    self.ingest()
                self.fresh(f"correlation-{index}")

    def test_direct_bundle_toctou_symlink_hardlink_mode_and_extra_are_rejected(self):
        snapshot = INGEST._DirectBundleSnapshot(
            self.fixture.root,
            uid=os.getuid(),
            gid=os.getgid(),
            forbidden_values=(self.fixture.secret,),
        )
        self.addCleanup(snapshot.close)
        for name in ("input-manifest.json", "summary.json"):
            snapshot.read_small(name)
        for name in (
            "http-client.raw.jsonl",
            "observer.raw.jsonl",
            "service-journal.raw.jsonl",
            "observer-journal-correlation.raw.jsonl",
        ):
            list(snapshot.iter_lines(name))
        target = self.fixture.root / "summary.json"
        replacement = self.fixture.root / "replacement"
        write_private(replacement, target.read_bytes())
        os.replace(replacement, target)
        with self.assertRaises(INGEST.GateIngestError):
            snapshot.seal()

        self.fresh("extra")
        write_private(self.fixture.root / "unexpected", b"unexpected\n")
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("mode")
        (self.fixture.root / "summary.json").chmod(0o640)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("hardlink")
        os.link(
            self.fixture.root / "summary.json",
            self.fixture.root.parent / "summary-hardlink",
        )
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

        self.fresh("symlink")
        summary = self.fixture.root / "summary.json"
        outside = self.fixture.root.parent / "outside-summary"
        summary.rename(outside)
        summary.symlink_to(outside)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_direct_secret_across_chunk_boundary_is_rejected(self):
        path = self.fixture.root / "summary.json"
        raw = path.read_bytes()
        padding = b" " * (INGEST.COPY_CHUNK_BYTES - 3 - len(raw))
        write_private(path, raw + padding + self.fixture.secret)
        with self.assertRaises(INGEST.GateIngestError):
            self.ingest()

    def test_actual_pilot_revalidates_only_when_all_bound_sources_match(self):
        if not DIRECT_PILOT.is_dir():
            self.skipTest("direct cancellation pilot is not present")
        expected_names = INGEST.DIRECT_CANCEL_FILES
        self.assertEqual({path.name for path in DIRECT_PILOT.iterdir()}, expected_names)
        self.assertEqual(DIRECT_PILOT.stat().st_mode & 0o777, 0o700)
        self.assertTrue(
            all(
                path.stat().st_mode & 0o777 == 0o600 and path.stat().st_nlink == 1
                for path in DIRECT_PILOT.iterdir()
            )
        )
        manifest = json.loads((DIRECT_PILOT / "input-manifest.json").read_bytes())
        manifest_hashes = {
            value["path"]: value["sha256"] for value in manifest["inputs"]
        }
        current_hashes = {
            "tools/run-sq8-direct-cancel-gate.py": digest(
                DIRECT_GATE_SOURCE.read_bytes()
            ),
            "tools/collect-sq8-openwebui-release.py": digest(
                COLLECTOR_SOURCE.read_bytes()
            ),
            "tools/sq8-openwebui-http-client.py": digest(
                HTTP_CLIENT_SOURCE.read_bytes()
            ),
        }
        mismatches = [
            path
            for path, current in current_hashes.items()
            if manifest_hashes.get(path) != current
        ]
        if mismatches:
            self.skipTest("pilot source binding differs from current source")
        summary = json.loads((DIRECT_PILOT / "summary.json").read_bytes())
        identity = summary["service_identity"]
        bindings = INGEST.DirectCancelInputBindings(
            gate_source=DIRECT_GATE_SOURCE,
            gate_source_sha256=current_hashes["tools/run-sq8-direct-cancel-gate.py"],
            collector_source=COLLECTOR_SOURCE,
            collector_source_sha256=current_hashes[
                "tools/collect-sq8-openwebui-release.py"
            ],
            http_client_source=HTTP_CLIENT_SOURCE,
            http_client_source_sha256=current_hashes[
                "tools/sq8-openwebui-http-client.py"
            ],
            http_image_id=summary["http_image_id"],
            docker_network_id=summary["docker_network_id"],
            service_unit=identity["unit"],
            service_user=identity["user"],
            boot_id=identity["boot_id"],
            control_group=identity["control_group"],
            gateway_pid=identity["gateway_pid"],
            gateway_starttime_ticks=identity["gateway_starttime_ticks"],
            worker_pid=identity["worker_pid"],
            worker_starttime_ticks=identity["worker_starttime_ticks"],
            restart_count=identity["n_restarts"],
            uid=identity["uid"],
            gid=identity["gid"],
            forbidden_values=(b"pilot-regression-secret-sentinel",),
        )
        result = INGEST.ingest_direct_cancel_bundle(DIRECT_PILOT, bindings)
        self.assertEqual(len(result.lifecycle_claims), 55)


def dataclass_replace(value, **changes):
    values = {
        field.name: getattr(value, field.name)
        for field in value.__dataclass_fields__.values()
    }
    values.update(changes)
    return type(value)(**values)


if __name__ == "__main__":
    unittest.main()
