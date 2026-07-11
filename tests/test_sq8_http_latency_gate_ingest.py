import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import queue
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = ROOT / "tools" / "sq8_http_latency_gate_ingest.py"
GATE_PATH = ROOT / "tools" / "run-sq8-http-latency-gate.py"
SECRET = b"latency-ingest-secret-value-0123456789"
IMAGE_ID = "sha256:" + "a" * 64
NETWORK_ID = "b" * 64
BOOT_ID = "c" * 32
GATEWAY_PID = 4101
WORKER_PID = 4102


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INGEST = load_module("sq8_http_latency_gate_ingest", INGEST_PATH)
GATE = load_module("run_sq8_http_latency_gate_for_ingest", GATE_PATH)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lifecycle_events(spec, request_id, completion_id, base, release_time):
    admitted_time = base + 3
    started_time = base + 4
    events = [
        {
            "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
            "event": "request_admitted",
            "observed_monotonic_ns": admitted_time,
            "request_id": request_id,
            "completion_id": completion_id,
            "stream": True,
            "prompt_tokens": spec.prompt_tokens,
            "max_completion_tokens": spec.max_tokens,
        },
        {
            "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
            "event": "request_started",
            "observed_monotonic_ns": started_time,
            "request_id": request_id,
            "completion_id": completion_id,
            "stream": True,
            "prompt_tokens": spec.prompt_tokens,
            "admit_to_start_ns": started_time - admitted_time,
        },
    ]
    processed = 0
    observed = started_time
    while processed < spec.prompt_tokens:
        processed = min(processed + 128, spec.prompt_tokens)
        observed += 1
        events.append(
            {
                "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
                "event": "request_progress",
                "observed_monotonic_ns": observed,
                "request_id": request_id,
                "completion_id": completion_id,
                "phase": "prefill",
                "processed_prompt_tokens": processed,
                "prompt_tokens": spec.prompt_tokens,
            }
        )
    observed += 1
    events.append(
        {
            "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
            "event": "request_first_token",
            "observed_monotonic_ns": observed,
            "request_id": request_id,
            "completion_id": completion_id,
            "stream": True,
            "completion_tokens": 1,
        }
    )
    if spec.workload == "ttft":
        cancel_time = release_time - 1
        events.append(
            {
                "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
                "event": "request_cancel_requested",
                "observed_monotonic_ns": cancel_time,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "reason": "client_disconnect",
                "admit_to_cancel_ns": cancel_time - admitted_time,
            }
        )
        outcome = "cancelled"
        cancel_reason = "client_disconnect"
        completion_tokens = 1
    else:
        outcome = "length"
        cancel_reason = None
        completion_tokens = 64
    events.append(
        {
            "schema_version": GATE.COL.LIFECYCLE_SCHEMA,
            "event": "request_released",
            "observed_monotonic_ns": release_time,
            "request_id": request_id,
            "completion_id": completion_id,
            "stream": True,
            "outcome": outcome,
            "cancel_reason": cancel_reason,
            "prompt_tokens": spec.prompt_tokens,
            "completion_tokens": completion_tokens,
            "reset_complete": True,
            "admit_to_start_ns": started_time - admitted_time,
            "start_to_release_ns": release_time - started_time,
            "admit_to_release_ns": release_time - admitted_time,
        }
    )
    return events


class SyntheticLatencyBundle:
    def __init__(self, root):
        self.root = root
        self.bundle = root / "latency-bundle"
        self.bundle.mkdir(mode=0o700)
        os.chmod(self.bundle, 0o700)
        self.epoch_path = root / "resource-restart-epoch.json"
        self.service_identity = {
            "unit": GATE.DIRECT.SERVICE_UNIT,
            "user": "homelab1",
            "uid": 1000,
            "gid": 1000,
            "control_group": "/system.slice/ullm-openai.service",
            "gateway_pid": GATEWAY_PID,
            "gateway_starttime_ticks": 51_001,
            "worker_pid": WORKER_PID,
            "worker_starttime_ticks": 51_002,
            "n_restarts": 2,
            "boot_id": BOOT_ID,
        }
        epoch = {
            "schema_version": GATE.EPOCH_SCHEMA,
            "phase": "resource_restart",
            "service_identity": self.service_identity,
        }
        self.epoch_path.write_bytes(GATE.compact_json(epoch))
        os.chmod(self.epoch_path, 0o600)
        self.fixtures = {
            fixture_id: GATE.load_fixture(
                ROOT
                / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
                / f"{fixture_id}.json",
                fixture_id,
            )
            for fixture_id in GATE.FIXTURE_ORDER
        }
        self._build()

    def bindings(self, **changes):
        values = {
            "gate_source": GATE_PATH,
            "gate_source_sha256": digest(GATE_PATH),
            "direct_source": ROOT / "tools/run-sq8-direct-cancel-gate.py",
            "direct_source_sha256": digest(
                ROOT / "tools/run-sq8-direct-cancel-gate.py"
            ),
            "collector_source": ROOT / "tools/collect-sq8-openwebui-release.py",
            "collector_source_sha256": digest(
                ROOT / "tools/collect-sq8-openwebui-release.py"
            ),
            "http_client_source": ROOT / "tools/sq8-openwebui-http-client.py",
            "http_client_source_sha256": digest(
                ROOT / "tools/sq8-openwebui-http-client.py"
            ),
            "restart_epoch_file": self.epoch_path,
            "restart_epoch_sha256": digest(self.epoch_path),
            "http_image_id": IMAGE_ID,
            "docker_network_id": NETWORK_ID,
            "service_unit": GATE.DIRECT.SERVICE_UNIT,
            "service_user": "homelab1",
            "boot_id": BOOT_ID,
            "control_group": "/system.slice/ullm-openai.service",
            "gateway_pid": GATEWAY_PID,
            "gateway_starttime_ticks": 51_001,
            "worker_pid": WORKER_PID,
            "worker_starttime_ticks": 51_002,
            "restart_count": 2,
            "uid": 1000,
            "gid": 1000,
            "forbidden_values": (SECRET,),
        }
        values.update(changes)
        return INGEST.LatencyGateInputBindings(**values)

    def _write(self, name, raw):
        path = self.bundle / name
        path.write_bytes(raw)
        os.chmod(path, 0o600)

    def _build(self):
        http = [
            GATE.compact_json(
                {
                    "schema_version": GATE.HTTP_EVENT_SCHEMA,
                    "event": "ready",
                    "observed_monotonic_ns": 1,
                }
            )
        ]
        observer = []
        journal = []
        correlations = []
        validator = GATE.LatencyRunValidator()
        samples = []
        clock = 1_000_000
        lifecycle_sequence = 0
        for spec in GATE.SCHEDULE:
            fixture = self.fixtures[spec.fixture_id]
            request_body = GATE.request_body(fixture, spec.max_tokens)
            request_id = f"req-latency-{spec.sequence:02d}"
            completion_id = f"chatcmpl-latency-{spec.sequence:02d}"
            sent = clock + 2
            request = {
                "schema_version": GATE.HTTP_EVENT_SCHEMA,
                "event": "http_request",
                "request_key": spec.case_id,
                "method": "POST",
                "target": GATE.HTTP_TARGET,
                "headers": {
                    "content_type": "application/json",
                    "content_length": len(request_body),
                    "authorization_mode": "valid_bearer",
                },
                "body_base64": base64.b64encode(request_body).decode("ascii"),
                "body_sha256": hashlib.sha256(request_body).hexdigest(),
                "body_bytes": len(request_body),
                "connect_completed_monotonic_ns": clock,
                "write_started_monotonic_ns": clock + 1,
                "last_body_byte_sent_monotonic_ns": sent,
            }
            start = {
                "schema_version": GATE.HTTP_EVENT_SCHEMA,
                "event": "http_response_start",
                "request_key": spec.case_id,
                "status": 200,
                "headers": [["Content-Type", "text/event-stream"]],
                "observed_monotonic_ns": clock + 5,
            }
            raw_chunks = []
            chunk_times = []
            if spec.workload == "ttft":
                value = {
                    "id": completion_id,
                    "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
                }
                raw_chunks.append(b"data: " + GATE.compact_json(value) + b"\n\n")
                chunk_times.append(clock + 20_000_000)
                outcome = "client_closed"
            else:
                for _index in range(64):
                    value = {
                        "id": completion_id,
                        "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
                    }
                    raw_chunks.append(b"data: " + GATE.compact_json(value) + b"\n\n")
                    chunk_times.append(clock + 20_000_000 + _index * 50_000_000)
                terminal = {
                    "id": completion_id,
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 64},
                }
                raw_chunks.append(
                    b"data: " + GATE.compact_json(terminal) + b"\n\ndata: [DONE]\n\n"
                )
                chunk_times.append(chunk_times[-1] + 1)
                outcome = "eof"
            body_digest = hashlib.sha256()
            body_size = 0
            chunks = []
            body_events = []
            for chunk_index, (raw, observed) in enumerate(
                zip(raw_chunks, chunk_times, strict=True)
            ):
                body_digest.update(raw)
                body_size += len(raw)
                chunks.append(GATE.TimedChunk(chunk_index, raw, observed))
                body_events.append(
                    {
                        "schema_version": GATE.HTTP_EVENT_SCHEMA,
                        "event": "http_body_chunk",
                        "request_key": spec.case_id,
                        "chunk_index": chunk_index,
                        "body_base64": base64.b64encode(raw).decode("ascii"),
                        "body_sha256": hashlib.sha256(raw).hexdigest(),
                        "body_bytes": len(raw),
                        "observed_monotonic_ns": observed,
                    }
                )
            response_end = chunk_times[-1] + 1
            end = {
                "schema_version": GATE.HTTP_EVENT_SCHEMA,
                "event": "http_response_end",
                "request_key": spec.case_id,
                "outcome": outcome,
                "error": None,
                "body_bytes": body_size,
                "body_sha256": body_digest.hexdigest(),
                "observed_monotonic_ns": response_end,
            }
            events = [request, start, *body_events, end]
            http.extend(GATE.compact_json(value) for value in events)
            items = GATE.parse_timed_sse(
                tuple(chunks), allow_incomplete=outcome == "client_closed"
            )
            observation = GATE.HttpObservation(
                status=200,
                outcome=outcome,
                request_sent_monotonic_ns=sent,
                response_start_monotonic_ns=clock + 5,
                response_end_monotonic_ns=response_end,
                body=b"".join(raw_chunks),
                chunks=tuple(chunks),
                items=items,
            )
            release_time = (
                chunk_times[0] + 2 if spec.workload == "ttft" else response_end + 2
            )
            events = lifecycle_events(
                spec, request_id, completion_id, clock, release_time
            )
            validator.begin(spec)
            for event in events:
                validator.consume(event)
                payload = GATE.compact_json(event)
                observer.append(payload)
                cursor = f"cursor-latency-{lifecycle_sequence:04d}"
                journal_monotonic = str((event["observed_monotonic_ns"] + 999) // 1000)
                journal_record = {
                    "__CURSOR": cursor,
                    "__MONOTONIC_TIMESTAMP": journal_monotonic,
                    "_BOOT_ID": BOOT_ID,
                    "_PID": str(GATEWAY_PID),
                    "_SYSTEMD_UNIT": GATE.DIRECT.SERVICE_UNIT,
                    "PRIORITY": "6",
                    "MESSAGE": payload.decode("ascii"),
                }
                journal.append(GATE.compact_json(journal_record))
                correlations.append(
                    GATE.compact_json(
                        {
                            "schema_version": GATE.GATE_SCHEMA,
                            "sequence": lifecycle_sequence,
                            "cursor": cursor,
                            "journal_monotonic_usec": journal_monotonic,
                            "journal_pid": str(GATEWAY_PID),
                            "observer_received_monotonic_ns": event[
                                "observed_monotonic_ns"
                            ]
                            + 100,
                            "observer_sender_pid": GATEWAY_PID,
                            "observer_sender_uid": 1000,
                            "observer_sender_gid": 1000,
                            "payload_sha256": hashlib.sha256(payload).hexdigest(),
                            "payload_bytes": len(payload),
                        }
                    )
                )
                lifecycle_sequence += 1
            samples.append(validator.complete(observation))
            clock = release_time + 1_000_000
        finalized, metrics = validator.finalize()
        self.assert_same(samples, finalized)
        http.append(
            GATE.compact_json(
                {
                    "schema_version": GATE.HTTP_EVENT_SCHEMA,
                    "event": "shutdown_complete",
                    "observed_monotonic_ns": clock,
                }
            )
        )
        self._write("http-client.raw.jsonl", b"\n".join(http) + b"\n")
        self._write("observer.raw.jsonl", b"\n".join(observer) + b"\n")
        self._write("service-journal.raw.jsonl", b"\n".join(journal) + b"\n")
        self._write(
            "observer-journal-correlation.raw.jsonl",
            b"\n".join(correlations) + b"\n",
        )
        self._write(
            "samples.raw.jsonl",
            b"\n".join(GATE.compact_json(value) for value in samples) + b"\n",
        )
        manifest = self._manifest()
        self._write("input-manifest.json", GATE.compact_json(manifest) + b"\n")
        summary = {
            "schema_version": GATE.GATE_SCHEMA,
            "record_type": "summary",
            "passed": True,
            "request_count": 72,
            "max_active": 1,
            "service_identity": self.service_identity,
            "resource_restart_epoch_sha256": digest(self.epoch_path),
            "http_image_id": IMAGE_ID,
            "docker_network_name": GATE.HTTP_NETWORK_NAME,
            "docker_network_id": NETWORK_ID,
            "observer_socket": os.fspath(GATE.OBSERVER_SOCKET),
            "observer_event_count": len(observer),
            "journal_correlation_count": len(correlations),
            "metrics": metrics,
            "artifacts": {
                name: self._artifact(name)
                for name in (
                    "http-client.raw.jsonl",
                    "observer.raw.jsonl",
                    "service-journal.raw.jsonl",
                    "observer-journal-correlation.raw.jsonl",
                    "samples.raw.jsonl",
                )
            },
        }
        self._write("summary.json", GATE.compact_json(summary) + b"\n")
        checksum_names = sorted(
            INGEST.EXPECTED_FILES - {"SHA256SUMS"},
            key=lambda value: value.encode("ascii"),
        )
        checksums = b"".join(
            f"{digest(self.bundle / name)}  {name}\n".encode("ascii")
            for name in checksum_names
        )
        self._write("SHA256SUMS", checksums)

    @staticmethod
    def assert_same(left, right):
        if left != right:
            raise AssertionError("synthetic sample construction differs")

    def _artifact(self, name):
        raw = (self.bundle / name).read_bytes()
        return {
            "bytes": len(raw),
            "lines": raw.count(b"\n"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _manifest(self):
        gate_raw = GATE_PATH.read_bytes()
        direct_raw = (ROOT / "tools/run-sq8-direct-cancel-gate.py").read_bytes()
        collector_raw = (ROOT / "tools/collect-sq8-openwebui-release.py").read_bytes()
        client_raw = (ROOT / "tools/sq8-openwebui-http-client.py").read_bytes()
        epoch_raw = self.epoch_path.read_bytes()
        return {
            "schema_version": GATE.GATE_SCHEMA,
            "record_type": "input_manifest",
            "inputs": [
                {
                    "path": "tools/run-sq8-http-latency-gate.py",
                    "bytes": len(gate_raw),
                    "sha256": hashlib.sha256(gate_raw).hexdigest(),
                },
                {
                    "path": "tools/run-sq8-direct-cancel-gate.py",
                    "bytes": len(direct_raw),
                    "sha256": hashlib.sha256(direct_raw).hexdigest(),
                },
                {
                    "path": "tools/collect-sq8-openwebui-release.py",
                    "bytes": len(collector_raw),
                    "sha256": hashlib.sha256(collector_raw).hexdigest(),
                },
                {
                    "path": "tools/sq8-openwebui-http-client.py",
                    "bytes": len(client_raw),
                    "sha256": hashlib.sha256(client_raw).hexdigest(),
                },
                {
                    "path": "resource-restart-epoch.json",
                    "bytes": len(epoch_raw),
                    "sha256": hashlib.sha256(epoch_raw).hexdigest(),
                },
                *[
                    {
                        "path": (
                            "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/"
                            f"{fixture_id}.json"
                        ),
                        "bytes": len(self.fixtures[fixture_id].raw),
                        "sha256": self.fixtures[fixture_id].sha256,
                    }
                    for fixture_id in GATE.FIXTURE_ORDER
                ],
            ],
            "schedule": [dataclasses.asdict(item) for item in GATE.SCHEDULE],
            "request_bodies": [
                {
                    "fixture_id": fixture_id,
                    "max_tokens": max_tokens,
                    "bytes": len(
                        GATE.request_body(self.fixtures[fixture_id], max_tokens)
                    ),
                    "sha256": hashlib.sha256(
                        GATE.request_body(self.fixtures[fixture_id], max_tokens)
                    ).hexdigest(),
                }
                for fixture_id, max_tokens in (
                    *((fixture_id, 512) for fixture_id in GATE.FIXTURE_ORDER),
                    ("exact-p0032", 64),
                )
            ],
        }


class LatencyGateIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = SyntheticLatencyBundle(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def ingest(self, bindings=None):
        return INGEST.ingest_latency_gate_bundle(
            self.fixture.bundle,
            bindings or self.fixture.bindings(),
        )

    @staticmethod
    def mutate_json(path, mutation):
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_bytes(GATE.compact_json(value) + b"\n")
        os.chmod(path, 0o600)

    @staticmethod
    def mutate_jsonl(path, line_index, mutation):
        lines = path.read_bytes().splitlines()
        value = json.loads(lines[line_index])
        mutation(value)
        lines[line_index] = GATE.compact_json(value)
        path.write_bytes(b"\n".join(lines) + b"\n")
        os.chmod(path, 0o600)

    def assert_rejected(self, callback):
        with self.assertRaises(INGEST.LatencyGateIngestError):
            callback()

    def test_positive_revalidates_72_requests_and_returns_campaign_material(self):
        result = self.ingest()
        requests = [
            value
            for value in result.http_records
            if value["record_type"] == "http_request"
        ]
        self.assertEqual(len(requests), 72)
        self.assertTrue(
            all(value["phase"] == "latency" for value in result.http_records)
        )
        self.assertGreater(len(result.lifecycle_claims), 72)
        self.assertTrue(
            all(value.phase == "latency" for value in result.lifecycle_claims)
        )
        self.assertEqual(result.derived_view["prefill_ttft"]["request_count"], 60)
        self.assertEqual(result.derived_view["decode64"]["request_count"], 12)
        self.assertEqual(
            result.derived_view["decode64"]["metrics"]["interval_count"], 630
        )
        derived = json.dumps(result.derived_view, sort_keys=True).encode("utf-8")
        self.assertNotIn(SECRET, derived)
        self.assertNotIn(b"chatcmpl-latency", derived)

    def test_all_claims_are_consumed_in_the_campaign_restart_epoch(self):
        result = self.ingest()
        campaign = sys.modules["sq8_openwebui_campaign"]

        class ControlledSource:
            def __init__(self):
                self.rows = queue.Queue()

            @staticmethod
            def open_after(_unit, _boot_id):
                return "latency-campaign-anchor"

            def read_next(self, timeout_usec):
                try:
                    return self.rows.get(timeout=min(timeout_usec / 1_000_000, 0.001))
                except queue.Empty:
                    return None

            @staticmethod
            def close():
                return None

        source = ControlledSource()
        capture = campaign.CampaignJournalCapture(
            self.root / "latency-campaign-journal.raw.jsonl",
            BOOT_ID,
            campaign.PidEpoch(GATEWAY_PID - 100, WORKER_PID - 100),
            scan_raw=lambda _raw, _label: None,
            source=source,
        )
        self.addCleanup(capture.abort)
        capture.start()
        capture.checkpoint("resource_normal", time.monotonic_ns() + 2_000_000_000)
        capture.arm_restart_transition()
        capture.confirm_restart_epoch(campaign.PidEpoch(GATEWAY_PID, WORKER_PID))
        for claim in result.lifecycle_claims:
            source.rows.put(claim.raw)
        claimed = capture.claim_bundle_records(
            result.lifecycle_claims, time.monotonic_ns() + 5_000_000_000
        )
        self.assertEqual(len(claimed), len(result.lifecycle_claims))
        self.assertTrue(all(value.phase == "latency" for value in claimed))
        self.assertEqual(claimed[0].case_id, GATE.SCHEDULE[0].case_id)
        self.assertEqual(claimed[-1].case_id, GATE.SCHEDULE[-1].case_id)

    def test_raw_http_mutation_is_rejected_before_producer_summary(self):
        path = self.fixture.bundle / "http-client.raw.jsonl"
        self.mutate_jsonl(
            path,
            1,
            lambda value: value.__setitem__("body_sha256", "0" * 64),
        )
        self.assert_rejected(self.ingest)

    def test_false_producer_verdict_is_rejected_after_raw_rederivation(self):
        self.mutate_json(
            self.fixture.bundle / "summary.json",
            lambda value: value.__setitem__("passed", False),
        )
        self.assert_rejected(self.ingest)

    def test_manifest_schedule_mutation_is_rejected(self):
        self.mutate_json(
            self.fixture.bundle / "input-manifest.json",
            lambda value: value["schedule"][0].__setitem__("sample_index", 2),
        )
        self.assert_rejected(self.ingest)

    def test_observer_journal_correlation_mutation_is_rejected(self):
        self.mutate_jsonl(
            self.fixture.bundle / "observer-journal-correlation.raw.jsonl",
            0,
            lambda value: value.__setitem__("observer_sender_pid", WORKER_PID),
        )
        self.assert_rejected(self.ingest)

    def test_journal_lifecycle_payload_mutation_is_rejected(self):
        def mutate(value):
            payload = json.loads(value["MESSAGE"])
            payload["completion_id"] = "chatcmpl-mutated"
            value["MESSAGE"] = GATE.compact_json(payload).decode("ascii")

        self.mutate_jsonl(self.fixture.bundle / "service-journal.raw.jsonl", 0, mutate)
        self.assert_rejected(self.ingest)

    def test_layout_and_mode_are_exact(self):
        extra = self.fixture.bundle / "extra.json"
        extra.write_bytes(b"{}\n")
        os.chmod(extra, 0o600)
        self.assert_rejected(self.ingest)
        extra.unlink()
        os.chmod(self.fixture.bundle / "summary.json", 0o644)
        self.assert_rejected(self.ingest)

    def test_symlink_and_hardlink_bundle_artifacts_are_rejected(self):
        for defect in ("symlink", "hardlink"):
            with self.subTest(defect=defect):
                base = self.root / defect
                base.mkdir()
                fixture = SyntheticLatencyBundle(base)
                target = fixture.bundle / "summary.json"
                if defect == "symlink":
                    outside = base / "outside-summary.json"
                    target.rename(outside)
                    target.symlink_to(outside)
                else:
                    os.link(target, base / "summary-second-link.json")
                with self.assertRaises(INGEST.LatencyGateIngestError):
                    INGEST.ingest_latency_gate_bundle(
                        fixture.bundle, fixture.bindings()
                    )

    def test_source_digest_binding_is_required(self):
        self.assert_rejected(
            lambda: self.ingest(self.fixture.bindings(gate_source_sha256="0" * 64))
        )

    def test_checksum_document_mutation_is_rejected(self):
        path = self.fixture.bundle / "SHA256SUMS"
        raw = path.read_bytes()
        replacement = b"0" if raw[:1] != b"0" else b"1"
        path.write_bytes(replacement + raw[1:])
        os.chmod(path, 0o600)
        self.assert_rejected(self.ingest)

    def test_secret_cleartext_is_rejected_while_streaming(self):
        path = self.fixture.bundle / "summary.json"
        path.write_bytes(path.read_bytes() + SECRET + b"\n")
        os.chmod(path, 0o600)
        self.assert_rejected(self.ingest)

    def test_toctou_mutation_after_validation_is_rejected_at_seal(self):
        original = INGEST._derived_view

        def mutate_before_seal(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.fixture.bundle / "summary.json"
            path.write_bytes(path.read_bytes() + b" ")
            os.chmod(path, 0o600)
            return result

        with mock.patch.object(INGEST, "_derived_view", side_effect=mutate_before_seal):
            self.assert_rejected(self.ingest)

    def test_bound_source_entry_replacement_is_rejected_at_seal(self):
        original = INGEST._derived_view

        def replace_source_before_seal(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.fixture.epoch_path
            replacement = self.root / "replacement-resource-restart-epoch.json"
            replacement.write_bytes(path.read_bytes())
            os.chmod(replacement, 0o600)
            os.replace(replacement, path)
            return result

        with mock.patch.object(
            INGEST, "_derived_view", side_effect=replace_source_before_seal
        ):
            self.assert_rejected(self.ingest)


if __name__ == "__main__":
    unittest.main()
