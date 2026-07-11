from __future__ import annotations

import base64
import copy
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run-sq8-api-contract-gate.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("run_sq8_api_contract_gate", GATE_PATH)


EXPECTED_CASES = (
    ("models-valid", "GET", "/v1/models", "valid_bearer", 200, None, None),
    (
        "models-missing-auth",
        "GET",
        "/v1/models",
        "missing",
        401,
        "invalid_api_key",
        None,
    ),
    (
        "models-invalid-auth",
        "GET",
        "/v1/models",
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
    ),
    (
        "models-query",
        "GET",
        "/v1/models?x=1",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
    ),
    (
        "chat-malformed-missing-auth",
        "POST",
        "/v1/chat/completions",
        "missing",
        401,
        "invalid_api_key",
        None,
    ),
    (
        "chat-invalid-auth",
        "POST",
        "/v1/chat/completions",
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
    ),
    (
        "chat-malformed-valid-auth",
        "POST",
        "/v1/chat/completions",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
    ),
    (
        "chat-duplicate-key",
        "POST",
        "/v1/chat/completions",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
    ),
    (
        "chat-unsupported-n",
        "POST",
        "/v1/chat/completions",
        "valid_bearer",
        400,
        "unsupported_parameter",
        "n",
    ),
    (
        "chat-missing-model",
        "POST",
        "/v1/chat/completions",
        "valid_bearer",
        404,
        "model_not_found",
        "model",
    ),
)


def compact(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def response_body(case):
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


def request_key(case, index):
    return f"api-contract-{index:02d}-{case.case_id}"


def valid_events(case, index, *, base=1_000_000):
    key = request_key(case, index)
    body = response_body(case)
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
            "body_sha256": hashlib.sha256(case.body).hexdigest(),
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
            "body_sha256": hashlib.sha256(body).hexdigest(),
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
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "observed_monotonic_ns": base + 5,
        },
    ]


def replace_response_body(events, body):
    changed = copy.deepcopy(events)
    chunk = changed[-2]
    end = changed[-1]
    chunk["body_base64"] = base64.b64encode(body).decode("ascii")
    chunk["body_sha256"] = hashlib.sha256(body).hexdigest()
    chunk["body_bytes"] = len(body)
    end["body_sha256"] = hashlib.sha256(body).hexdigest()
    end["body_bytes"] = len(body)
    for pair in changed[1]["headers"]:
        if pair[0].lower() == "content-length":
            pair[1] = str(len(body))
    return changed


def observation(case, index, *, base):
    return GATE.parse_http_events(
        case, request_key(case, index), valid_events(case, index, base=base)
    )


class FakeRuntime:
    def __init__(self, *, mutate=None, drift_at=None, extra_lifecycle_at=None):
        self.mutate = mutate
        self.drift_at = drift_at
        self.extra_lifecycle_at = extra_lifecycle_at
        self.identity_checks = 0
        self.requests = []
        self.quiet_checks = []

    def require_identity(self):
        self.identity_checks += 1
        if self.identity_checks == self.drift_at:
            raise GATE.GateError("service identity drift")

    def request(self, case, key):
        index = len(self.requests) + 1
        self.requests.append((case.case_id, key))
        result = observation(case, index, base=index * 100)
        if self.mutate is not None:
            result = self.mutate(index, case, result)
        return result

    def quiet(self, label):
        self.quiet_checks.append(label)
        if len(self.quiet_checks) == self.extra_lifecycle_at:
            raise GATE.GateError("unexpected worker lifecycle")


class FakeWriter:
    def __init__(self):
        self.lines = []

    def write(self, raw, _label):
        self.lines.append(raw)


class FakeObserver:
    def __init__(self, *, fail_empty=False):
        self.fail_empty = fail_empty

    def require_empty(self):
        if self.fail_empty:
            raise GATE.GateError("observer is not empty")


class FakeSource:
    def __init__(self):
        self.cursor = "anchor"


class FakeJournal:
    def __init__(self, records):
        self.records = list(records)
        self.source = FakeSource()

    def poll(self):
        return None


class ScheduleTests(unittest.TestCase):
    def test_frozen_schedule_exactly_matches_the_ten_non_gpu_cases(self):
        self.assertEqual(len(GATE.FROZEN_SCHEDULE), 10)
        actual = tuple(
            (
                case.case_id,
                case.method,
                case.target,
                case.authorization_mode,
                case.expected_status,
                case.expected_code,
                case.expected_param,
            )
            for case in GATE.FROZEN_SCHEDULE
        )
        self.assertEqual(actual, EXPECTED_CASES)
        GATE.validate_schedule(GATE.FROZEN_SCHEDULE)
        self.assertEqual(GATE.FROZEN_SCHEDULE[0].body, b"")
        self.assertEqual(GATE.FROZEN_SCHEDULE[4].body, GATE.MALFORMED_BODY)
        self.assertEqual(GATE.FROZEN_SCHEDULE[4].authorization_mode, "missing")
        self.assertIn(b'"n":2', GATE.FROZEN_SCHEDULE[8].body)
        self.assertIn(b'"model":"missing"', GATE.FROZEN_SCHEDULE[9].body)

    def test_schedule_deletion_reordering_duplication_and_body_change_fail(self):
        cases = GATE.FROZEN_SCHEDULE
        mutations = (
            cases[:-1],
            tuple(reversed(cases)),
            cases[:-1] + (cases[-2],),
            (dataclasses.replace(cases[0], body=b"changed"), *cases[1:]),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(GATE.GateError, "schedule"):
                    GATE.validate_schedule(mutation)

    def test_request_commands_preserve_method_target_body_and_auth_without_key(self):
        secret = b"api-key-must-not-appear"
        for index, case in enumerate(GATE.FROZEN_SCHEDULE, start=1):
            command = GATE.build_request_command(case, request_key(case, index))
            self.assertEqual(command["method"], case.method)
            self.assertEqual(command["target"], case.target)
            self.assertEqual(command["authorization_mode"], case.authorization_mode)
            self.assertEqual(base64.b64decode(command["body_base64"]), case.body)
            self.assertNotIn(secret, compact(command))


class HttpEventTests(unittest.TestCase):
    def test_all_cases_reconstruct_raw_http_and_validate_exact_semantics(self):
        prior_end = -1
        summaries = []
        for index, case in enumerate(GATE.FROZEN_SCHEDULE, start=1):
            events = valid_events(case, index, base=index * 100)
            result = GATE.parse_http_events(
                case,
                request_key(case, index),
                events,
                previous_response_end_ns=prior_end,
            )
            summary = GATE.validate_case_observation(case, result, index)
            summaries.append(summary)
            prior_end = result.response_end_monotonic_ns
            self.assertEqual(result.request_body, case.body)
            self.assertEqual(result.response_body, response_body(case))
            self.assertEqual(
                summary["response_body_sha256"],
                hashlib.sha256(result.response_body).hexdigest(),
            )
        self.assertEqual(
            [item["case_id"] for item in summaries],
            [case.case_id for case in GATE.FROZEN_SCHEDULE],
        )
        self.assertEqual(summaries[0]["error"], None)
        self.assertEqual(summaries[8]["error"]["param"], "n")
        self.assertNotIn(GATE.INVALID_KEY_MESSAGE.encode("ascii"), compact(summaries))

    def test_request_method_target_auth_body_hash_and_order_mutations_fail(self):
        case = GATE.FROZEN_SCHEDULE[5]
        base_events = valid_events(case, 6)
        mutations = []
        for field, value in (
            ("method", "GET"),
            ("target", "/v1/models"),
            ("request_key", "wrong-key"),
            ("body_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(base_events)
            changed[0][field] = value
            mutations.append(changed)
        changed = copy.deepcopy(base_events)
        changed[0]["headers"]["authorization_mode"] = "valid_bearer"
        mutations.append(changed)
        changed = copy.deepcopy(base_events)
        changed[0]["body_base64"] = base64.b64encode(b"changed").decode("ascii")
        changed[0]["body_bytes"] = 7
        changed[0]["body_sha256"] = hashlib.sha256(b"changed").hexdigest()
        changed[0]["headers"]["content_length"] = 7
        mutations.append(changed)
        mutations.append([base_events[1], base_events[0], *base_events[2:]])
        for events in mutations:
            with self.subTest(events=events):
                with self.assertRaises(GATE.GateError):
                    GATE.parse_http_events(case, request_key(case, 6), events)

    def test_chunk_hash_size_timestamp_aggregate_and_terminal_mutations_fail(self):
        case = GATE.FROZEN_SCHEDULE[6]
        base_events = valid_events(case, 7)
        mutations = []
        for target, field, value in (
            (-2, "body_sha256", "0" * 64),
            (-2, "chunk_index", 1),
            (-2, "observed_monotonic_ns", 0),
            (-1, "body_bytes", 1),
            (-1, "body_sha256", "0" * 64),
            (-1, "outcome", "error"),
            (-1, "error", "diagnostic"),
        ):
            changed = copy.deepcopy(base_events)
            changed[target][field] = value
            mutations.append(changed)
        extra = copy.deepcopy(base_events)
        extra.insert(-1, copy.deepcopy(extra[-2]))
        mutations.append(extra)
        for events in mutations:
            with self.subTest(events=events):
                with self.assertRaises(GATE.GateError):
                    GATE.parse_http_events(case, request_key(case, 7), events)

    def test_content_type_and_www_authenticate_are_exact(self):
        cases = (GATE.FROZEN_SCHEDULE[0], GATE.FROZEN_SCHEDULE[1])
        for index, case in enumerate(cases, start=1):
            base_events = valid_events(case, index)
            result = GATE.parse_http_events(case, request_key(case, index), base_events)
            GATE.validate_case_observation(case, result, index)
            mutations = []
            no_content = copy.deepcopy(base_events)
            no_content[1]["headers"] = [
                pair for pair in no_content[1]["headers"] if pair[0] != "content-type"
            ]
            mutations.append(no_content)
            wrong_content = copy.deepcopy(base_events)
            for pair in wrong_content[1]["headers"]:
                if pair[0] == "content-type":
                    pair[1] = "text/plain"
            mutations.append(wrong_content)
            content_with_parameter = copy.deepcopy(base_events)
            for pair in content_with_parameter[1]["headers"]:
                if pair[0] == "content-type":
                    pair[1] = "application/json; charset=utf-8"
            mutations.append(content_with_parameter)
            wrong_auth = copy.deepcopy(base_events)
            if case.expected_status == 401:
                for pair in wrong_auth[1]["headers"]:
                    if pair[0] == "www-authenticate":
                        pair[1] = "Basic"
            else:
                wrong_auth[1]["headers"].append(["www-authenticate", "Bearer"])
            mutations.append(wrong_auth)
            wrong_length = copy.deepcopy(base_events)
            for pair in wrong_length[1]["headers"]:
                if pair[0] == "content-length":
                    pair[1] = "1"
            mutations.append(wrong_length)
            retry_after = copy.deepcopy(base_events)
            retry_after[1]["headers"].append(["retry-after", "1"])
            mutations.append(retry_after)
            transfer_encoding = copy.deepcopy(base_events)
            transfer_encoding[1]["headers"].append(["transfer-encoding", "chunked"])
            mutations.append(transfer_encoding)
            for events in mutations:
                changed = GATE.parse_http_events(case, request_key(case, index), events)
                with self.assertRaises(GATE.GateError):
                    GATE.validate_case_observation(case, changed, index)

    def test_error_envelope_status_type_code_param_and_extra_fields_are_exact(self):
        case = GATE.FROZEN_SCHEDULE[8]
        base_events = valid_events(case, 9)
        base_value = json.loads(response_body(case))
        mutations = []
        for field, value in (
            ("type", "server_error"),
            ("code", "invalid_request_error"),
            ("param", None),
            ("message", ""),
            ("message", "A different but non-empty public diagnostic."),
        ):
            changed = copy.deepcopy(base_value)
            changed["error"][field] = value
            mutations.append(replace_response_body(base_events, compact(changed)))
        changed = copy.deepcopy(base_value)
        changed["error"]["extra"] = True
        mutations.append(replace_response_body(base_events, compact(changed)))
        wrong_status = copy.deepcopy(base_events)
        wrong_status[1]["status"] = 422
        mutations.append(wrong_status)
        for events in mutations:
            result = GATE.parse_http_events(case, request_key(case, 9), events)
            with self.assertRaises(GATE.GateError):
                GATE.validate_case_observation(case, result, 9)

    def test_model_list_is_exact_single_model(self):
        case = GATE.FROZEN_SCHEDULE[0]
        events = valid_events(case, 1)
        value = json.loads(response_body(case))
        for mutation in (
            {**value, "extra": True},
            {**value, "data": []},
            {
                **value,
                "data": [
                    *value["data"],
                    {"id": "other", "object": "model", "owned_by": "ullm"},
                ],
            },
        ):
            changed = replace_response_body(events, compact(mutation))
            result = GATE.parse_http_events(case, request_key(case, 1), changed)
            with self.assertRaisesRegex(GATE.GateError, "model list"):
                GATE.validate_case_observation(case, result, 1)

    def test_duplicate_json_response_keys_and_non_finite_values_fail(self):
        case = GATE.FROZEN_SCHEDULE[1]
        for body in (
            b'{"error":{"message":"x","message":"y","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}',
            b'{"error":{"message":"x","type":"invalid_request_error","param":null,"code":NaN}}',
        ):
            result = GATE.parse_http_events(
                case,
                request_key(case, 2),
                replace_response_body(valid_events(case, 2), body),
            )
            with self.assertRaises(GATE.GateError):
                GATE.validate_case_observation(case, result, 2)


class RunnerAndQuietTests(unittest.TestCase):
    def test_runner_is_serial_and_checks_identity_and_quiet_after_every_case(self):
        runtime = FakeRuntime()
        runner = GATE.ApiContractRunner(runtime)
        results = runner.run()
        self.assertEqual(len(results), 10)
        self.assertEqual(runner.max_active, 1)
        self.assertEqual(runtime.identity_checks, 11)
        self.assertEqual(
            runtime.quiet_checks, [case.case_id for case in GATE.FROZEN_SCHEDULE]
        )
        self.assertEqual([item[0] for item in runtime.requests], runtime.quiet_checks)

    def test_runner_rejects_service_identity_drift(self):
        runtime = FakeRuntime(drift_at=4)
        with self.assertRaisesRegex(GATE.GateError, "identity drift"):
            GATE.ApiContractRunner(runtime).run()

    def test_runner_rejects_extra_worker_lifecycle(self):
        runtime = FakeRuntime(extra_lifecycle_at=3)
        with self.assertRaisesRegex(GATE.GateError, "worker lifecycle"):
            GATE.ApiContractRunner(runtime).run()

    def test_runner_rejects_response_order_overlap(self):
        def overlap(index, _case, result):
            if index != 2:
                return result
            return dataclasses.replace(result, connect_completed_monotonic_ns=0)

        with self.assertRaisesRegex(GATE.GateError, "prior response"):
            GATE.ApiContractRunner(FakeRuntime(mutate=overlap)).run()

    def test_authoritative_journal_lifecycle_is_rejected_even_without_observer(self):
        event = {
            "schema_version": GATE.DIRECT.COL.LIFECYCLE_SCHEMA,
            "event": "request_admitted",
            "observed_monotonic_ns": 1000,
            "request_id": "req-unexpected",
            "completion_id": "chatcmpl-unexpected",
            "stream": True,
            "prompt_tokens": 1,
            "max_completion_tokens": 1,
        }
        record = compact(
            {
                "__CURSOR": "cursor-unexpected",
                "__MONOTONIC_TIMESTAMP": "1",
                "_BOOT_ID": "5" * 32,
                "_PID": "1200",
                "_SYSTEMD_UNIT": GATE.DIRECT.SERVICE_UNIT,
                "PRIORITY": "6",
                "MESSAGE": "INFO:     " + compact(event).decode("ascii"),
            }
        )
        runtime = GATE.ProductionRuntime(
            object(), object(), FakeObserver(), FakeJournal([record]), FakeWriter()
        )
        with self.assertRaisesRegex(GATE.GateError, "worker lifecycle"):
            runtime._poll_journal_lifecycle()

    def test_observer_nonempty_is_rejected_before_quiet_record(self):
        writer = FakeWriter()
        runtime = GATE.ProductionRuntime(
            object(), object(), FakeObserver(fail_empty=True), FakeJournal([]), writer
        )
        with mock.patch.object(GATE.time, "sleep", return_value=None):
            with self.assertRaisesRegex(GATE.GateError, "observer"):
                runtime.quiet("case")
        self.assertEqual(writer.lines, [])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guard = GATE.DIRECT.COL.SecretGuard(b"api-contract-artifact-secret")

    def test_source_snapshot_rejects_same_path_content_or_identity_change(self):
        path = self.root / "source.py"
        path.write_bytes(b"original source\n")
        raw, identity = GATE._snapshot(path, "test source", 1024)
        GATE.verify_snapshot(path, "test source", raw, identity)
        path.write_bytes(b"modified source\n")
        with self.assertRaises(GATE.GateError):
            GATE.verify_snapshot(path, "test source", raw, identity)

    def test_sha256sums_is_sorted_exact_and_detects_artifact_mutation(self):
        GATE.write_file(self.root / "b.json", b"b\n", self.guard, "b")
        GATE.write_file(self.root / "a.json", b"a\n", self.guard, "a")
        document = GATE.write_sha256sums(self.root, ("b.json", "a.json"), self.guard)
        lines = document.decode("ascii").splitlines()
        self.assertTrue(lines[0].endswith("  a.json"))
        self.assertTrue(lines[1].endswith("  b.json"))
        GATE.verify_sha256sums(self.root, ("b.json", "a.json"), document)
        (self.root / "b.json").write_bytes(b"changed\n")
        with self.assertRaisesRegex(GATE.GateError, "digest"):
            GATE.verify_sha256sums(self.root, ("b.json", "a.json"), document)

    def test_secret_is_rejected_before_artifact_creation(self):
        target = self.root / "secret.json"
        with self.assertRaises(Exception):
            GATE.write_file(
                target,
                b'{"value":"api-contract-artifact-secret"}\n',
                self.guard,
                "secret artifact",
            )
        self.assertFalse(target.exists())

    def test_atomic_bundle_is_fresh_and_exclusive(self):
        final = self.root / "bundle"
        transaction = GATE.DIRECT.AtomicRunDirectory(final)
        (transaction.stage / "evidence").write_bytes(b"sealed\n")
        transaction.publish()
        self.assertEqual((final / "evidence").read_bytes(), b"sealed\n")
        with self.assertRaises(Exception):
            GATE.DIRECT.AtomicRunDirectory(final)

    def test_locked_publication_pins_mode_link_count_inode_and_hash(self):
        final = self.root / "locked-bundle"
        transaction = GATE.DIRECT.AtomicRunDirectory(final)
        GATE.write_file(
            transaction.stage / "evidence.json", b"sealed\n", self.guard, "evidence"
        )
        lock = GATE.LockedStage(transaction.stage, ("evidence.json",), self.guard)
        try:
            lock.open()
            GATE.publish_locked(transaction, lock)
            self.assertTrue(transaction.published)
            self.assertEqual((final / "evidence.json").read_bytes(), b"sealed\n")
            metadata = (final / "evidence.json").lstat()
            self.assertEqual(metadata.st_mode & 0o777, 0o600)
            self.assertEqual(metadata.st_nlink, 1)
        finally:
            lock.close()
            if final.exists():
                final.chmod(0o700)

    def test_locked_publication_rejects_mode_hardlink_and_same_size_mutation(self):
        for defect in ("mode", "hardlink", "mutation"):
            with self.subTest(defect=defect):
                final = self.root / f"bundle-{defect}"
                transaction = GATE.DIRECT.AtomicRunDirectory(final)
                artifact = transaction.stage / "evidence.json"
                GATE.write_file(artifact, b"original\n", self.guard, "evidence")
                if defect == "mode":
                    artifact.chmod(0o644)
                elif defect == "hardlink":
                    os_link = transaction.stage / "second-link"
                    os_link.hardlink_to(artifact)
                lock = GATE.LockedStage(
                    transaction.stage,
                    ("evidence.json", "second-link")
                    if defect == "hardlink"
                    else ("evidence.json",),
                    self.guard,
                )
                try:
                    if defect in {"mode", "hardlink"}:
                        with self.assertRaises(GATE.GateError):
                            lock.open()
                    else:
                        lock.open()
                        artifact.write_bytes(b"modified\n")
                        with self.assertRaises(GATE.GateError):
                            GATE.publish_locked(transaction, lock)
                        self.assertFalse(final.exists())
                finally:
                    lock.close()
                    transaction.abort()

    def test_locked_stage_rejects_symlink_and_extra_or_missing_layout(self):
        for defect in ("symlink", "extra", "missing"):
            with self.subTest(defect=defect):
                transaction = GATE.DIRECT.AtomicRunDirectory(
                    self.root / f"layout-{defect}"
                )
                artifact = transaction.stage / "evidence.json"
                GATE.write_file(artifact, b"sealed\n", self.guard, "evidence")
                names = ("evidence.json",)
                if defect == "symlink":
                    artifact.unlink()
                    artifact.symlink_to("target")
                elif defect == "extra":
                    GATE.write_file(
                        transaction.stage / "extra.json",
                        b"extra\n",
                        self.guard,
                        "extra",
                    )
                else:
                    names = ("evidence.json", "missing.json")
                lock = GATE.LockedStage(transaction.stage, names, self.guard)
                try:
                    with self.assertRaises(Exception):
                        lock.open()
                finally:
                    lock.close()
                    transaction.abort()

    def test_locked_publication_never_replaces_a_racing_destination(self):
        final = self.root / "raced-bundle"
        transaction = GATE.DIRECT.AtomicRunDirectory(final)
        GATE.write_file(
            transaction.stage / "evidence.json", b"sealed\n", self.guard, "evidence"
        )
        lock = GATE.LockedStage(transaction.stage, ("evidence.json",), self.guard)
        try:
            lock.open()
            final.mkdir()
            (final / "competitor").write_bytes(b"keep\n")
            with self.assertRaises(Exception):
                GATE.publish_locked(transaction, lock)
            self.assertEqual((final / "competitor").read_bytes(), b"keep\n")
        finally:
            lock.close()
            transaction.abort()

    def test_input_manifest_contains_sources_and_bodies_but_no_credential_identity(
        self,
    ):
        gateway_sources = {
            path: f"source:{path}".encode("ascii")
            for path in GATE.GATEWAY_SOURCE_RELATIVES
        }
        manifest = GATE.build_input_manifest(
            b"gate source", b"client source", gateway_sources
        )
        raw = compact(manifest)
        self.assertEqual(len(manifest["inputs"]), 7)
        self.assertEqual(len(manifest["request_bodies"]), 10)
        self.assertNotIn(b"api_key", raw.lower())
        self.assertNotIn(b"credential", raw.lower())
        self.assertNotIn(b"api-contract-artifact-secret", raw)

    def test_write_file_refuses_to_replace_an_existing_artifact(self):
        path = self.root / "fresh.json"
        GATE.write_file(path, b"first\n", self.guard, "fresh")
        with self.assertRaises(GATE.GateError):
            GATE.write_file(path, b"second\n", self.guard, "fresh")
        self.assertEqual(path.read_bytes(), b"first\n")


class SourceIdentityTests(unittest.TestCase):
    def test_committed_http_client_hash_and_stable_support_snapshots_match(self):
        client = (ROOT / "tools" / "sq8-openwebui-http-client.py").read_bytes()
        self.assertEqual(hashlib.sha256(client).hexdigest(), GATE.HTTP_CLIENT_SHA256)
        self.assertEqual(
            client and GATE.DIRECT.HTTP_CLIENT_SHA256, GATE.HTTP_CLIENT_SHA256
        )
        self.assertEqual(
            hashlib.sha256(GATE.DIRECT_SUPPORT_RAW).hexdigest(),
            hashlib.sha256(
                (ROOT / "tools" / "run-sq8-direct-cancel-gate.py").read_bytes()
            ).hexdigest(),
        )

    def test_unexpected_exception_uses_one_fixed_credential_free_diagnostic(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        argv = [
            "--output-dir",
            "/tmp/not-created",
            "--api-key-file",
            "/secret/credential/path",
            "--http-image-id",
            "sha256:" + "a" * 64,
            "--docker-network-id",
            "b" * 64,
        ]
        with (
            mock.patch.object(
                GATE, "execute", side_effect=RuntimeError("/secret/host/path")
            ),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(GATE.main(argv), 2)
        self.assertEqual(stderr.getvalue(), "SQ8 API contract gate failed\n")
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("secret", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
