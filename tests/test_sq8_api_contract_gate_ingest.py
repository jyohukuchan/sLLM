from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run-sq8-api-contract-gate.py"
ADAPTER_PATH = ROOT / "tools" / "sq8_api_contract_gate_ingest.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("test_sq8_api_contract_gate_ingest_gate", GATE_PATH)
ADAPTER = load_module("test_sq8_api_contract_gate_ingest_adapter", ADAPTER_PATH)


def compact(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def document(value: Any) -> bytes:
    return compact(value) + b"\n"


def json_lines(values: list[dict[str, Any]]) -> bytes:
    return b"".join(document(value) for value in values)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def response_body(case: Any) -> bytes:
    if case.expect_models:
        return compact(
            {
                "object": "list",
                "data": [{"id": GATE.MODEL_ID, "object": "model", "owned_by": "ullm"}],
            }
        )
    return compact(
        {
            "error": {
                "message": case.expected_message,
                "type": "invalid_request_error",
                "param": case.expected_param,
                "code": case.expected_code,
            }
        }
    )


def request_key(case: Any, index: int) -> str:
    return f"api-contract-{index:02d}-{case.case_id}"


def http_case_events(case: Any, index: int) -> list[dict[str, Any]]:
    key = request_key(case, index)
    body = response_body(case)
    base = index * 10_000
    response_headers = [
        ["date", "Sat, 11 Jul 2026 00:00:00 GMT"],
        ["content-type", "application/json"],
        ["content-length", str(len(body))],
    ]
    if case.expected_status == 401:
        response_headers.append(["www-authenticate", "Bearer"])
    return [
        {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_request",
            "request_key": key,
            "method": case.method,
            "target": case.target,
            "headers": {
                "content_type": "application/json",
                "content_length": len(case.body),
                "authorization_mode": case.authorization_mode,
            },
            "body_base64": base64.b64encode(case.body).decode("ascii"),
            "body_sha256": sha256(case.body),
            "body_bytes": len(case.body),
            "connect_completed_monotonic_ns": base,
            "write_started_monotonic_ns": base + 1,
            "last_body_byte_sent_monotonic_ns": base + 2,
        },
        {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_start",
            "request_key": key,
            "status": case.expected_status,
            "headers": response_headers,
            "observed_monotonic_ns": base + 3,
        },
        {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_body_chunk",
            "request_key": key,
            "chunk_index": 0,
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_sha256": sha256(body),
            "body_bytes": len(body),
            "observed_monotonic_ns": base + 4,
        },
        {
            "schema_version": GATE.HTTP_EVENT_SCHEMA,
            "event": "http_response_end",
            "request_key": key,
            "outcome": "eof",
            "error": None,
            "body_bytes": len(body),
            "body_sha256": sha256(body),
            "observed_monotonic_ns": base + 5,
        },
    ]


def source_bindings(*, forbidden_values: tuple[bytes, ...] = ()) -> Any:
    sources = {
        "gate_source": GATE_PATH,
        "direct_source": ROOT / "tools" / "run-sq8-direct-cancel-gate.py",
        "collector_source": ROOT / "tools" / "collect-sq8-openwebui-release.py",
        "http_client_source": ROOT / "tools" / "sq8-openwebui-http-client.py",
        "gateway_app_source": (
            ROOT
            / "services"
            / "openai-gateway"
            / "src"
            / "ullm_openai_gateway"
            / "app.py"
        ),
        "gateway_errors_source": (
            ROOT
            / "services"
            / "openai-gateway"
            / "src"
            / "ullm_openai_gateway"
            / "errors.py"
        ),
        "gateway_schemas_source": (
            ROOT
            / "services"
            / "openai-gateway"
            / "src"
            / "ullm_openai_gateway"
            / "schemas.py"
        ),
    }
    values: dict[str, Any] = {}
    for field, path in sources.items():
        values[field] = path
        values[f"{field.removesuffix('_source')}_source_sha256"] = sha256(
            path.read_bytes()
        )
    return ADAPTER.ApiContractInputBindings(
        **values,
        http_image_id="sha256:" + "a" * 64,
        docker_network_id="b" * 64,
        service_unit="ullm-openai.service",
        service_user="homelab1",
        boot_id="5" * 32,
        control_group="/system.slice/ullm-openai.service",
        gateway_pid=1200,
        gateway_starttime_ticks=10_000,
        worker_pid=1201,
        worker_starttime_ticks=10_001,
        restart_count=2,
        uid=os.getuid(),
        gid=os.getgid(),
        forbidden_values=forbidden_values,
    )


class GateBundle:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name)
        self.root = self.base / "api-contract"
        self.root.mkdir(mode=0o700)
        self.bindings = source_bindings()
        self.http_events: list[dict[str, Any]] = [
            {
                "schema_version": GATE.HTTP_EVENT_SCHEMA,
                "event": "ready",
                "observed_monotonic_ns": 1_000,
            }
        ]
        self.case_summaries: list[dict[str, Any]] = []
        for index, case in enumerate(GATE.FROZEN_SCHEDULE, start=1):
            events = http_case_events(case, index)
            self.http_events.extend(events)
            observation = GATE.parse_http_events(
                case,
                request_key(case, index),
                events,
                previous_response_end_ns=(index - 1) * 10_000 + 5 if index > 1 else -1,
            )
            self.case_summaries.append(
                GATE.validate_case_observation(case, observation, index)
            )
        self.http_events.append(
            {
                "schema_version": GATE.HTTP_EVENT_SCHEMA,
                "event": "shutdown_complete",
                "observed_monotonic_ns": 110_000,
            }
        )
        self.journal_records = [
            {
                "__CURSOR": f"cursor-{index:02d}",
                "__MONOTONIC_TIMESTAMP": str(200_000 + index),
                "_BOOT_ID": self.bindings.boot_id,
                "_PID": str(self.bindings.gateway_pid),
                "_SYSTEMD_UNIT": self.bindings.service_unit,
                "PRIORITY": "6",
                "MESSAGE": f"gateway access record {index:02d}",
            }
            for index in range(1, 14)
        ]
        labels = [case.case_id for case in GATE.FROZEN_SCHEDULE] + [
            "http-client-shutdown",
            "post-observer-close",
            "final-readiness-and-identity",
        ]
        self.quiet_records = [
            {
                "schema_version": GATE.GATE_SCHEMA,
                "record_type": "lifecycle_quiet_check",
                "sequence": sequence,
                "label": label,
                "checked_monotonic_ns": 300_000 + sequence,
                "observer_open": sequence <= 10,
                "observer_event_count": 0,
                "new_journal_record_count": 1,
                "journal_record_count": sequence + 1,
                "journal_cursor": f"cursor-{sequence + 1:02d}",
            }
            for sequence, label in enumerate(labels)
        ]
        self._write_initial()

    def __enter__(self) -> GateBundle:
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.root.exists() and not self.root.is_symlink():
            os.chmod(self.root, 0o700)
        self._temporary.cleanup()

    def _write(self, name: str, raw: bytes) -> None:
        os.chmod(self.root, 0o700)
        path = self.root / name
        path.write_bytes(raw)
        os.chmod(path, 0o600)

    def _raw_artifact_metadata(self, name: str) -> dict[str, Any]:
        raw = (self.root / name).read_bytes()
        return {"bytes": len(raw), "lines": raw.count(b"\n"), "sha256": sha256(raw)}

    def _write_initial(self) -> None:
        self._write("http-client.raw.jsonl", json_lines(self.http_events))
        self._write("service-journal.raw.jsonl", json_lines(self.journal_records))
        self._write("lifecycle-quiet.raw.jsonl", json_lines(self.quiet_records))
        gateway_sources = {
            relative: (ROOT / relative).read_bytes()
            for relative in GATE.GATEWAY_SOURCE_RELATIVES
        }
        manifest = GATE.build_input_manifest(
            GATE_PATH.read_bytes(),
            (ROOT / "tools" / "sq8-openwebui-http-client.py").read_bytes(),
            gateway_sources,
        )
        self._write("input-manifest.json", document(manifest))
        summary = {
            "schema_version": GATE.GATE_SCHEMA,
            "record_type": "summary",
            "model_id": GATE.MODEL_ID,
            "request_count": 10,
            "max_active": 1,
            "service_identity": {
                "unit": self.bindings.service_unit,
                "user": self.bindings.service_user,
                "uid": self.bindings.uid,
                "gid": self.bindings.gid,
                "control_group": self.bindings.control_group,
                "gateway_pid": self.bindings.gateway_pid,
                "gateway_starttime_ticks": self.bindings.gateway_starttime_ticks,
                "worker_pid": self.bindings.worker_pid,
                "worker_starttime_ticks": self.bindings.worker_starttime_ticks,
                "n_restarts": self.bindings.restart_count,
                "boot_id": self.bindings.boot_id,
            },
            "http_image_id": self.bindings.http_image_id,
            "docker_network_name": GATE.DIRECT.HTTP_NETWORK_NAME,
            "docker_network_id": self.bindings.docker_network_id,
            "observer_socket": os.fspath(GATE.OBSERVER_SOCKET),
            "observer_event_count": 0,
            "lifecycle_event_count": 0,
            "quiet_check_count": 13,
            "cases": self.case_summaries,
            "artifacts": {
                name: self._raw_artifact_metadata(name)
                for name in (
                    "http-client.raw.jsonl",
                    "service-journal.raw.jsonl",
                    "lifecycle-quiet.raw.jsonl",
                )
            },
        }
        self._write("summary.json", document(summary))
        self.rebuild_checksums()
        self.seal_layout()

    def seal_layout(self) -> None:
        for name in ADAPTER.EXPECTED_FILES:
            path = self.root / name
            if path.exists() and not path.is_symlink():
                os.chmod(path, 0o600)
        os.chmod(self.root, 0o500)

    def rebuild_checksums(self) -> None:
        lines = []
        for name in ADAPTER.CHECKSUM_INPUTS:
            lines.append(f"{sha256((self.root / name).read_bytes())}  {name}\n")
        self._write("SHA256SUMS", "".join(lines).encode("ascii"))

    def mutate_document(
        self, name: str, mutation: Callable[[dict[str, Any]], None]
    ) -> None:
        value = json.loads((self.root / name).read_bytes())
        mutation(value)
        self._write(name, document(value))
        self.rebuild_checksums()
        self.seal_layout()

    def replace_raw(
        self, name: str, raw: bytes, *, refresh_summary: bool = True
    ) -> None:
        self._write(name, raw)
        if refresh_summary and name in {
            "http-client.raw.jsonl",
            "service-journal.raw.jsonl",
            "lifecycle-quiet.raw.jsonl",
        }:
            summary = json.loads((self.root / "summary.json").read_bytes())
            summary["artifacts"][name] = self._raw_artifact_metadata(name)
            self._write("summary.json", document(summary))
        self.rebuild_checksums()
        self.seal_layout()


class ApiContractGateIngestTests(unittest.TestCase):
    def ingest(self, fixture: GateBundle, bindings: Any | None = None) -> Any:
        return ADAPTER.ingest_api_contract_bundle(
            fixture.root, fixture.bindings if bindings is None else bindings
        )

    def assert_rejected(self, fixture: GateBundle, bindings: Any | None = None) -> None:
        with self.assertRaises(ADAPTER.ApiContractIngestError):
            self.ingest(fixture, bindings)

    def test_valid_bundle_returns_explicit_records_view_and_final_cursor(self) -> None:
        with GateBundle() as fixture:
            result = self.ingest(fixture)

        self.assertIsInstance(result, ADAPTER.ApiContractIngestResult)
        self.assertEqual(len(result.http_records), 40)
        self.assertEqual(result.final_journal_cursor, "cursor-13")
        self.assertEqual(result.derived_view["case_count"], 10)
        self.assertEqual(result.derived_view["http_record_count"], 40)
        self.assertEqual(result.derived_view["journal_record_count"], 13)
        self.assertEqual(
            {record["record_type"] for record in result.http_records},
            {
                "http_request",
                "http_response_start",
                "http_body_chunk",
                "http_response_end",
            },
        )
        for record in result.http_records:
            self.assertEqual(record["phase"], "api_contract")
            self.assertIn(
                record["case_id"], {case.case_id for case in GATE.FROZEN_SCHEDULE}
            )
        requests = [
            record
            for record in result.http_records
            if record["record_type"] == "http_request"
        ]
        for index, (record, case) in enumerate(
            zip(requests, GATE.FROZEN_SCHEDULE, strict=True), start=1
        ):
            self.assertEqual(record["fields"]["request_index"], index)
            self.assertEqual(record["fields"]["request_key"], request_key(case, index))

    def test_derived_view_excludes_cleartext_messages_prompts_and_paths(self) -> None:
        secret = b"adapter-bound-secret"
        with GateBundle() as fixture:
            bindings = source_bindings(forbidden_values=(secret,))
            result = self.ingest(fixture, bindings)
            raw = compact(result.derived_view)
        for message in (
            GATE.INVALID_KEY_MESSAGE,
            GATE.QUERY_MESSAGE,
            GATE.INVALID_JSON_MESSAGE,
            GATE.UNSUPPORTED_MESSAGE,
            GATE.MODEL_NOT_FOUND_MESSAGE,
            "API contract preflight",
        ):
            self.assertNotIn(message.encode("ascii"), raw)
        self.assertNotIn(secret, raw)
        for path, _digest in ADAPTER._source_specs(bindings).values():
            self.assertNotIn(os.fspath(path).encode(), raw)

    def test_root_mode_and_extra_layout_are_rejected(self) -> None:
        for defect in ("root-mode", "extra"):
            with self.subTest(defect=defect), GateBundle() as fixture:
                os.chmod(fixture.root, 0o700)
                if defect == "extra":
                    (fixture.root / "extra").write_bytes(b"extra\n")
                    os.chmod(fixture.root / "extra", 0o600)
                    os.chmod(fixture.root, 0o500)
                self.assert_rejected(fixture)

    def test_file_mode_symlink_and_hardlink_are_rejected(self) -> None:
        for defect in ("file-mode", "symlink", "hardlink"):
            with self.subTest(defect=defect), GateBundle() as fixture:
                os.chmod(fixture.root, 0o700)
                target = fixture.root / "summary.json"
                if defect == "file-mode":
                    os.chmod(target, 0o644)
                else:
                    original = target.read_bytes()
                    target.unlink()
                    outside = fixture.base / "outside-summary.json"
                    outside.write_bytes(original)
                    os.chmod(outside, 0o600)
                    if defect == "symlink":
                        target.symlink_to(outside)
                    else:
                        os.link(outside, target)
                os.chmod(fixture.root, 0o500)
                self.assert_rejected(fixture)

    def test_checksum_document_mutation_is_rejected(self) -> None:
        with GateBundle() as fixture:
            fixture._write("SHA256SUMS", b"0" * 64 + b"  summary.json\n")
            fixture.seal_layout()
            self.assert_rejected(fixture)

    def test_summary_case_and_producer_count_mutations_are_rejected(self) -> None:
        mutations = (
            lambda value: value["cases"][0].__setitem__("status", 201),
            lambda value: value.__setitem__("request_count", 9),
            lambda value: value.__setitem__("max_active", True),
            lambda value: value.__setitem__("observer_event_count", False),
            lambda value: value.__setitem__("passed", True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), GateBundle() as fixture:
                fixture.mutate_document("summary.json", mutation)
                self.assert_rejected(fixture)

    def test_manifest_source_binding_mutation_is_rejected(self) -> None:
        with GateBundle() as fixture:
            fixture.mutate_document(
                "input-manifest.json",
                lambda value: value["inputs"][0].__setitem__("sha256", "0" * 64),
            )
            self.assert_rejected(fixture)

    def test_bound_source_hash_mismatch_is_rejected(self) -> None:
        with GateBundle() as fixture:
            bindings = dataclasses.replace(
                fixture.bindings, gateway_app_source_sha256="0" * 64
            )
            self.assert_rejected(fixture, bindings)

    def test_reordered_case_schedule_is_rejected_from_raw_http(self) -> None:
        with GateBundle() as fixture:
            lines = fixture.http_events
            changed = [lines[0], *lines[5:9], *lines[1:5], *lines[9:]]
            fixture.replace_raw("http-client.raw.jsonl", json_lines(changed))
            self.assert_rejected(fixture)

    def test_raw_request_auth_mutation_is_rejected_even_with_refreshed_hashes(
        self,
    ) -> None:
        with GateBundle() as fixture:
            changed = copy.deepcopy(fixture.http_events)
            changed[1]["headers"]["authorization_mode"] = "missing"
            fixture.replace_raw("http-client.raw.jsonl", json_lines(changed))
            self.assert_rejected(fixture)

    def test_lifecycle_injection_is_rejected_even_with_refreshed_hashes(self) -> None:
        with GateBundle() as fixture:
            records = copy.deepcopy(fixture.journal_records)
            lifecycle = {
                "schema_version": GATE.DIRECT.COL.LIFECYCLE_SCHEMA,
                "event": "request_admitted",
                "observed_monotonic_ns": 1,
                "request_id": "req-injected",
                "completion_id": "chatcmpl-injected",
                "stream": True,
                "prompt_tokens": 1,
                "max_completion_tokens": 1,
            }
            records[2]["MESSAGE"] = "INFO:     " + compact(lifecycle).decode("ascii")
            fixture.replace_raw("service-journal.raw.jsonl", json_lines(records))
            self.assert_rejected(fixture)

    def test_journal_identity_cursor_and_monotonic_mutations_are_rejected(self) -> None:
        def wrong_pid(records: list[dict[str, Any]]) -> None:
            records[0]["_PID"] = "1201"

        def duplicate_cursor(records: list[dict[str, Any]]) -> None:
            records[1]["__CURSOR"] = records[0]["__CURSOR"]

        def regressed_monotonic(records: list[dict[str, Any]]) -> None:
            records[1]["__MONOTONIC_TIMESTAMP"] = "1"

        for mutation in (wrong_pid, duplicate_cursor, regressed_monotonic):
            with self.subTest(mutation=mutation), GateBundle() as fixture:
                records = copy.deepcopy(fixture.journal_records)
                mutation(records)
                fixture.replace_raw("service-journal.raw.jsonl", json_lines(records))
                self.assert_rejected(fixture)

    def test_quiet_order_observer_and_count_mutations_are_rejected(self) -> None:
        def wrong_label(records: list[dict[str, Any]]) -> None:
            records[1]["label"] = records[0]["label"]

        def wrong_observer(records: list[dict[str, Any]]) -> None:
            records[11]["observer_open"] = True

        def wrong_count(records: list[dict[str, Any]]) -> None:
            records[5]["new_journal_record_count"] = 0

        def boolean_event_count(records: list[dict[str, Any]]) -> None:
            records[5]["observer_event_count"] = False

        for mutation in (
            wrong_label,
            wrong_observer,
            wrong_count,
            boolean_event_count,
        ):
            with self.subTest(mutation=mutation), GateBundle() as fixture:
                records = copy.deepcopy(fixture.quiet_records)
                mutation(records)
                fixture.replace_raw("lifecycle-quiet.raw.jsonl", json_lines(records))
                self.assert_rejected(fixture)

    def test_bundle_snapshot_detects_entry_replacement_at_seal(self) -> None:
        with GateBundle() as fixture:
            snapshot = ADAPTER.BundleSnapshot(
                fixture.root,
                uid=fixture.bindings.uid,
                gid=fixture.bindings.gid,
                forbidden_values=(),
            )
            try:
                for name in (
                    "http-client.raw.jsonl",
                    "service-journal.raw.jsonl",
                    "lifecycle-quiet.raw.jsonl",
                ):
                    list(snapshot.iter_lines(name))
                for name in ("input-manifest.json", "summary.json", "SHA256SUMS"):
                    snapshot.read_small(name)
                replacement = fixture.base / "replacement-summary.json"
                replacement.write_bytes((fixture.root / "summary.json").read_bytes())
                os.chmod(replacement, 0o600)
                os.chmod(fixture.root, 0o700)
                os.replace(replacement, fixture.root / "summary.json")
                os.chmod(fixture.root, 0o500)
                with self.assertRaises(ADAPTER.ApiContractIngestError):
                    snapshot.seal()
            finally:
                snapshot.close()

    def test_bound_source_detects_entry_replacement_at_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            path = parent / "source.py"
            raw = b"value = 1\n"
            path.write_bytes(raw)
            source = ADAPTER._StableSource(path, "test source", 1024, sha256(raw), ())
            try:
                replacement = parent / "replacement.py"
                replacement.write_bytes(raw)
                os.replace(replacement, path)
                with self.assertRaises(ADAPTER.ApiContractIngestError):
                    source.seal()
            finally:
                source.close()

    def test_secret_split_across_stream_chunks_is_rejected(self) -> None:
        secret = b"split-boundary-secret"
        with GateBundle() as fixture:
            record = copy.deepcopy(fixture.journal_records[0])
            record["MESSAGE"] = ""
            empty = compact(record)
            marker = b'"MESSAGE":"'
            prefix = empty.index(marker) + len(marker)
            padding = ADAPTER.COPY_CHUNK_BYTES - 2 - prefix
            self.assertGreater(padding, 0)
            record["MESSAGE"] = "A" * padding + secret.decode("ascii")
            raw = document(record)
            self.assertEqual(raw.index(secret), ADAPTER.COPY_CHUNK_BYTES - 2)
            fixture.replace_raw("service-journal.raw.jsonl", raw)
            bindings = source_bindings(forbidden_values=(secret,))
            self.assert_rejected(fixture, bindings)


if __name__ == "__main__":
    unittest.main()
