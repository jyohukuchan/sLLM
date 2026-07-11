from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import os
import pwd
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run-openwebui-failure-gate.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "run_openwebui_failure_gate", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


SCREENSHOT = b"\x89PNG\r\n\x1a\nformal-failure"
SCREENSHOT_SHA = digest(SCREENSHOT)
TARGET = {
    "chat_id_utf8_bytes": 8,
    "chat_id_sha256": digest("chat-one"),
    "message_id_utf8_bytes": 11,
    "message_id_sha256": digest("message-one"),
}
RECOVERY = {
    "chat_id_utf8_bytes": 8,
    "chat_id_sha256": digest("chat-one"),
    "message_id_utf8_bytes": 11,
    "message_id_sha256": digest("message-two"),
}
ACTION_STARTS = (100, 120, 140, 160, 200, 280, 300, 320, 340)


def action(index: int) -> dict:
    name = TOOL.FINAL_ACTIONS[index]
    selectors = (
        None,
        "body",
        "#chat-input",
        ".chat-assistant",
        ".chat-assistant",
        "#chat-input",
        "#chat-input",
        ".chat-assistant",
        "#chat-input",
    )
    input_sha = None
    if index == 0:
        input_sha = "a" * 64
    elif index == 1:
        input_sha = digest(TOOL.MODEL_ID)
    elif index == 2:
        input_sha = digest(TOOL.FAILURE_PROMPT)
    elif index == 6:
        input_sha = digest(TOOL.RECOVERY_PROMPT)
    enabled = True if name in {"submit_chat", "wait_ready"} else None
    carries_text = name in {"wait_visible", "wait_failed"} or index == 8
    return {
        "browser_case": TOOL.BROWSER_CASE,
        "action_index": index,
        "action": name,
        "selector": selectors[index],
        "input_sha256": input_sha,
        "started_monotonic_ns": str(ACTION_STARTS[index]),
        "completed_monotonic_ns": str(ACTION_STARTS[index] + 5),
        "result": {
            "visible": True,
            "enabled": enabled,
            "text_utf8_bytes": 12 if carries_text else None,
            "text_sha256": "b" * 64 if carries_text else None,
        },
        "screenshot_file": f"browser/{TOOL.SCREENSHOT_NAME}" if index == 4 else None,
        "screenshot_sha256": SCREENSHOT_SHA if index == 4 else None,
    }


def socket_event(
    sequence: int,
    target: str,
    kind: str,
    observed: int,
    *,
    content: bool = False,
    error: bool = False,
    done: bool = False,
) -> dict:
    return {
        "sequence": sequence,
        "observed_monotonic_ns": str(observed),
        "correlation_target": target,
        "type": kind,
        "done": done,
        "has_error": error,
        "content_utf8_bytes": 7 if content else 0,
        "content_sha256": "c" * 64 if content else None,
    }


def clear_control(stage: str, nonce: str, path: str) -> dict:
    raw = TOOL.control_content(stage, nonce)
    return {
        "control_schema": TOOL.CONTROL_SCHEMA,
        "control_stage": stage,
        "control_file": path,
        "nonce": nonce,
        "content_utf8_bytes": len(raw),
        "content_sha256": digest(raw),
        "timeout_ms": 180_000,
    }


def redacted_control(
    stage: str, nonce: str, path: str, requested: int, observed: int
) -> dict:
    raw = TOOL.control_content(stage, nonce)
    return {
        "control_schema": TOOL.CONTROL_SCHEMA,
        "control_stage": stage,
        "control_file_utf8_bytes": len(path.encode()),
        "control_file_sha256": digest(path),
        "nonce_sha256": digest(nonce),
        "content_utf8_bytes": len(raw),
        "content_sha256": digest(raw),
        "requested_monotonic_ns": str(requested),
        "observed_monotonic_ns": str(observed),
    }


KILL_NONCE = "d" * 64
RECOVERY_NONCE = "e" * 64


def kill_interim_fixture() -> dict:
    return {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_worker_kill_wait",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "180",
        "browser_actions": [action(index) for index in range(4)],
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "submit_completed_monotonic_ns": "145",
            "visible_completed_monotonic_ns": "165",
            "pre_fault_done_count": 0,
            "pre_fault_error_count": 0,
            "pre_fault_cancel_count": 0,
        },
        "socket_events": [
            socket_event(0, "failure_target", "chat:completion", 150, content=True)
        ],
        "page_error_count": 0,
        "worker_killed_control": clear_control(
            "worker_killed", KILL_NONCE, TOOL.KILL_CONTROL_CONTAINER_PATH
        ),
    }


def recovery_interim_fixture() -> dict:
    return {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_gateway_recovery_wait",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "240",
        "browser_actions": [action(index) for index in range(5)],
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "error_first_observed_monotonic_ns": "210",
            "cancel_first_observed_monotonic_ns": "220",
            "error_event_count": 1,
            "cancel_event_count": 1,
            "done_after_fault_count": 0,
            "content_after_error_count": 0,
        },
        "socket_events": [
            socket_event(0, "failure_target", "chat:completion", 150, content=True),
            socket_event(1, "failure_target", "chat:completion", 210, error=True),
            socket_event(2, "failure_target", "chat:tasks:cancel", 220),
        ],
        "page_error_count": 0,
        "worker_killed_control": redacted_control(
            "worker_killed",
            KILL_NONCE,
            TOOL.KILL_CONTROL_CONTAINER_PATH,
            180,
            190,
        ),
        "gateway_recovered_control": clear_control(
            "gateway_recovered",
            RECOVERY_NONCE,
            TOOL.RECOVERY_CONTROL_CONTAINER_PATH,
        ),
    }


def final_fixture() -> dict:
    return {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_smoke",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "360",
        "browser_actions": [action(index) for index in range(9)],
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "error_first_observed_monotonic_ns": "210",
            "cancel_first_observed_monotonic_ns": "220",
            "error_event_count": 1,
            "cancel_event_count": 1,
            "done_after_fault_count": 0,
            "content_after_error_count": 0,
            "recovery": {
                **copy.deepcopy(RECOVERY),
                "submit_completed_monotonic_ns": "305",
                "done_observed_monotonic_ns": "335",
                "done_event_count": 1,
                "cancel_event_count": 0,
                "error_event_count": 0,
            },
        },
        "page_error_count": 0,
        "page_errors": [],
        "socket_events": [
            socket_event(0, "failure_target", "chat:completion", 150, content=True),
            socket_event(1, "failure_target", "chat:completion", 210, error=True),
            socket_event(2, "failure_target", "chat:tasks:cancel", 220),
            socket_event(3, "recovery_target", "chat:completion", 330, content=True),
            socket_event(4, "recovery_target", "chat:completion", 335, done=True),
        ],
        "controls": {
            "worker_killed": redacted_control(
                "worker_killed",
                KILL_NONCE,
                TOOL.KILL_CONTROL_CONTAINER_PATH,
                180,
                190,
            ),
            "gateway_recovered": redacted_control(
                "gateway_recovered",
                RECOVERY_NONCE,
                TOOL.RECOVERY_CONTROL_CONTAINER_PATH,
                240,
                290,
            ),
        },
        "screenshot": {
            "screenshot_file": f"browser/{TOOL.SCREENSHOT_NAME}",
            "screenshot_bytes": len(SCREENSHOT),
            "screenshot_sha256": SCREENSHOT_SHA,
        },
    }


def lifecycle(name: str, timestamp: int, request: str, completion: str, **fields):
    return {
        "schema_version": TOOL.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": timestamp,
        "request_id": request,
        "completion_id": completion,
        **fields,
    }


def lifecycle_records() -> list:
    old_request, old_completion = "req-old", "chatcmpl-old"
    new_request, new_completion = "req-new", "chatcmpl-new"
    old = [
        lifecycle(
            "request_admitted",
            100,
            old_request,
            old_completion,
            stream=True,
            prompt_tokens=44,
            max_completion_tokens=512,
        ),
        lifecycle(
            "request_started",
            110,
            old_request,
            old_completion,
            stream=True,
            prompt_tokens=44,
            admit_to_start_ns=10,
        ),
        lifecycle(
            "request_progress",
            120,
            old_request,
            old_completion,
            phase="prefill",
            processed_prompt_tokens=44,
            prompt_tokens=44,
        ),
        lifecycle(
            "request_first_token",
            130,
            old_request,
            old_completion,
            stream=True,
            completion_tokens=1,
        ),
        lifecycle(
            "worker_fatal",
            200,
            old_request,
            old_completion,
            reason="unexpected worker stdout EOF",
            admit_to_fatal_ns=100,
        ),
    ]
    new = [
        lifecycle(
            "request_admitted",
            310,
            new_request,
            new_completion,
            stream=True,
            prompt_tokens=80,
            max_completion_tokens=512,
        ),
        lifecycle(
            "request_started",
            312,
            new_request,
            new_completion,
            stream=True,
            prompt_tokens=80,
            admit_to_start_ns=2,
        ),
        lifecycle(
            "request_progress",
            315,
            new_request,
            new_completion,
            phase="prefill",
            processed_prompt_tokens=80,
            prompt_tokens=80,
        ),
        lifecycle(
            "request_first_token",
            320,
            new_request,
            new_completion,
            stream=True,
            completion_tokens=1,
        ),
        lifecycle(
            "request_released",
            334,
            new_request,
            new_completion,
            stream=True,
            outcome="stop",
            cancel_reason=None,
            prompt_tokens=80,
            completion_tokens=4,
            reset_complete=True,
            admit_to_start_ns=2,
            start_to_release_ns=22,
            admit_to_release_ns=24,
        ),
    ]
    records = []
    for index, event in enumerate(old + new):
        raw = TOOL.compact_json(event)
        records.append(
            TOOL.JournalLifecycle(
                cursor=f"cursor-{index}",
                journal_monotonic_usec=index + 1,
                journal_pid=1000 if index < len(old) else 2000,
                raw=raw,
                event=event,
            )
        )
    return records


def browser_final_for_lifecycle():
    return TOOL.BrowserFinal(
        target=copy.deepcopy(TARGET),
        recovery=copy.deepcopy(RECOVERY),
        visible_completed_ns=165,
        error_observed_ns=210,
        cancel_observed_ns=220,
        recovery_submit_started_ns=300,
        recovery_done_ns=335,
        first_target_content_ns=150,
        first_recovery_content_ns=330,
        screenshot_sha256=SCREENSHOT_SHA,
        action_count=9,
        socket_event_count=5,
        recovery_control_observed_ns=290,
    )


class BrowserEvidenceTests(unittest.TestCase):
    def validate_all(self, root: Path):
        screenshot = root / TOOL.SCREENSHOT_NAME
        summary = root / TOOL.BROWSER_SUMMARY_NAME
        screenshot.write_bytes(SCREENSHOT)
        kill = kill_interim_fixture()
        kill_raw = TOOL.compact_json(kill)
        kill_result = TOOL.validate_kill_interim(
            kill,
            kill_raw,
            TOOL.SecretGuard([]),
            expected_timeout_ms=180_000,
        )
        recovery = recovery_interim_fixture()
        recovery_raw = TOOL.compact_json(recovery)
        recovery_result = TOOL.validate_recovery_interim(
            recovery,
            recovery_raw,
            TOOL.SecretGuard([]),
            kill=kill_result,
            expected_timeout_ms=180_000,
            screenshot_path=screenshot,
        )
        final = final_fixture()
        final_raw = TOOL.compact_json(final)
        summary.write_bytes(final_raw + b"\n")
        return TOOL.validate_final_browser(
            final,
            final_raw,
            summary,
            screenshot,
            TOOL.SecretGuard([]),
            recovery_interim=recovery_result,
        )

    def test_three_browser_records_validate_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.validate_all(Path(temporary))
        self.assertEqual(result.action_count, 9)
        self.assertEqual(result.socket_event_count, 5)
        self.assertEqual(result.recovery_control_observed_ns, 290)

    def test_browser_record_prefixes_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screenshot = root / TOOL.SCREENSHOT_NAME
            screenshot.write_bytes(SCREENSHOT)
            kill_value = kill_interim_fixture()
            kill = TOOL.validate_kill_interim(
                kill_value,
                TOOL.compact_json(kill_value),
                TOOL.SecretGuard([]),
                expected_timeout_ms=180_000,
            )
            recovery_value = recovery_interim_fixture()
            recovery_value["browser_actions"][3]["result"]["text_sha256"] = "d" * 64
            with self.assertRaisesRegex(TOOL.FailureGateError, "action prefix changed"):
                TOOL.validate_recovery_interim(
                    recovery_value,
                    TOOL.compact_json(recovery_value),
                    TOOL.SecretGuard([]),
                    kill=kill,
                    expected_timeout_ms=180_000,
                    screenshot_path=screenshot,
                )

            recovery_value = recovery_interim_fixture()
            recovery_value["socket_events"][0]["content_sha256"] = "d" * 64
            with self.assertRaisesRegex(TOOL.FailureGateError, "socket prefix changed"):
                TOOL.validate_recovery_interim(
                    recovery_value,
                    TOOL.compact_json(recovery_value),
                    TOOL.SecretGuard([]),
                    kill=kill,
                    expected_timeout_ms=180_000,
                    screenshot_path=screenshot,
                )

    def test_boolean_values_cannot_substitute_for_integer_evidence(self):
        value = kill_interim_fixture()
        value["page_error_count"] = False
        with self.assertRaisesRegex(TOOL.FailureGateError, "bounded integer"):
            TOOL.validate_kill_interim(
                value,
                TOOL.compact_json(value),
                TOOL.SecretGuard([]),
                expected_timeout_ms=180_000,
            )
        value = kill_interim_fixture()
        value["browser_actions"][0]["action_index"] = False
        with self.assertRaisesRegex(TOOL.FailureGateError, "bounded integer"):
            TOOL.validate_kill_interim(
                value,
                TOOL.compact_json(value),
                TOOL.SecretGuard([]),
                expected_timeout_ms=180_000,
            )
        events = kill_interim_fixture()["socket_events"]
        events[0]["sequence"] = False
        with self.assertRaisesRegex(TOOL.FailureGateError, "bounded integer"):
            TOOL.validate_socket_events(
                events, allow_recovery=False, require_failure=False
            )

    def test_mutated_target_control_and_screenshot_are_rejected(self):
        kill = kill_interim_fixture()
        kill["worker_killed_control"]["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(TOOL.FailureGateError, "control evidence"):
            TOOL.validate_kill_interim(
                kill,
                TOOL.compact_json(kill),
                TOOL.SecretGuard([]),
                expected_timeout_ms=180_000,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / TOOL.SCREENSHOT_NAME).write_bytes(b"not-png")
            prior = TOOL.validate_kill_interim(
                kill_interim_fixture(),
                TOOL.compact_json(kill_interim_fixture()),
                TOOL.SecretGuard([]),
                expected_timeout_ms=180_000,
            )
            value = recovery_interim_fixture()
            with self.assertRaisesRegex(TOOL.FailureGateError, "PNG"):
                TOOL.validate_recovery_interim(
                    value,
                    TOOL.compact_json(value),
                    TOOL.SecretGuard([]),
                    kill=prior,
                    expected_timeout_ms=180_000,
                    screenshot_path=root / TOOL.SCREENSHOT_NAME,
                )

    def test_content_after_error_or_recovery_done_is_rejected(self):
        events = final_fixture()["socket_events"]
        events.insert(
            3,
            socket_event(3, "failure_target", "chat:completion", 225, content=True),
        )
        for index, event in enumerate(events):
            event["sequence"] = index
        with self.assertRaisesRegex(TOOL.FailureGateError, "follows provider error"):
            TOOL.validate_socket_events(
                events, allow_recovery=True, require_failure=True
            )
        events = final_fixture()["socket_events"]
        events.append(
            socket_event(5, "recovery_target", "chat:completion", 336, content=True)
        )
        with self.assertRaisesRegex(TOOL.FailureGateError, "follows normal completion"):
            TOOL.validate_socket_events(
                events, allow_recovery=True, require_failure=True
            )
        events = final_fixture()["socket_events"]
        events.insert(2, socket_event(2, "failure_target", "chat:completion", 215))
        for index, event in enumerate(events):
            event["sequence"] = index
        with self.assertRaisesRegex(TOOL.FailureGateError, "completion follows"):
            TOOL.validate_socket_events(
                events, allow_recovery=True, require_failure=True
            )


class LifecycleTests(unittest.TestCase):
    def test_failure_and_recovery_epochs_validate(self):
        result = TOOL.validate_failure_lifecycle(
            lifecycle_records(),
            initial_gateway_pid=1000,
            recovered_gateway_pid=2000,
            fault_started_ns=180,
            fault_completed_ns=181,
            browser=browser_final_for_lifecycle(),
        )
        self.assertEqual(result.lifecycle_count, 10)
        self.assertEqual(result.worker_fatal_ns, 200)
        self.assertEqual(result.recovery_released_ns, 334)
        browser = dataclass_replace(
            browser_final_for_lifecycle(), first_target_content_ns=129
        )
        with self.assertRaisesRegex(TOOL.FailureGateError, "fields or ordering"):
            TOOL.validate_failure_lifecycle(
                lifecycle_records(),
                initial_gateway_pid=1000,
                recovered_gateway_pid=2000,
                fault_started_ns=180,
                fault_completed_ns=181,
                browser=browser,
            )

    def test_foreign_pid_release_arithmetic_and_extra_fatal_are_rejected(self):
        records = lifecycle_records()
        records[3] = dataclass_replace(records[3], journal_pid=3000)
        with self.assertRaisesRegex(TOOL.FailureGateError, "PID"):
            TOOL.validate_failure_lifecycle(
                records,
                initial_gateway_pid=1000,
                recovered_gateway_pid=2000,
                fault_started_ns=180,
                fault_completed_ns=181,
                browser=browser_final_for_lifecycle(),
            )

        records = lifecycle_records()
        records[-1].event["completion_tokens"] = 513
        with self.assertRaisesRegex(TOOL.FailureGateError, "recovery lifecycle fields"):
            TOOL.validate_failure_lifecycle(
                records,
                initial_gateway_pid=1000,
                recovered_gateway_pid=2000,
                fault_started_ns=180,
                fault_completed_ns=181,
                browser=browser_final_for_lifecycle(),
            )

        records = lifecycle_records()
        records[-1].event["admit_to_start_ns"] = 3
        records[-1].event["admit_to_release_ns"] = 25
        with self.assertRaisesRegex(TOOL.FailureGateError, "recovery lifecycle fields"):
            TOOL.validate_failure_lifecycle(
                records,
                initial_gateway_pid=1000,
                recovered_gateway_pid=2000,
                fault_started_ns=180,
                fault_completed_ns=181,
                browser=browser_final_for_lifecycle(),
            )
        release = lifecycle_records()[-1].event
        release["admit_to_release_ns"] += 1
        with self.assertRaisesRegex(TOOL.FailureGateError, "duration"):
            TOOL.validate_lifecycle_payload(TOOL.compact_json(release))
        records = lifecycle_records()
        records.insert(5, copy.deepcopy(records[4]))
        records[5] = dataclass_replace(records[5], cursor="extra")
        with self.assertRaisesRegex(TOOL.FailureGateError, "sequence"):
            TOOL.validate_failure_lifecycle(
                records,
                initial_gateway_pid=1000,
                recovered_gateway_pid=2000,
                fault_started_ns=180,
                fault_completed_ns=181,
                browser=browser_final_for_lifecycle(),
            )

    def test_prefault_requires_exact_active_sequence(self):
        records = lifecycle_records()[:4]
        TOOL.validate_prefault_lifecycle(records, 1000)
        with self.assertRaisesRegex(TOOL.FailureGateError, "sequence"):
            TOOL.validate_prefault_lifecycle(records[:-1], 1000)
        changed = lifecycle_records()[:4]
        changed[2].event["processed_prompt_tokens"] = 43
        with self.assertRaisesRegex(TOOL.FailureGateError, "fields differ"):
            TOOL.validate_prefault_lifecycle(changed, 1000)


def dataclass_replace(value, **changes):
    return type(value)(
        **{
            field.name: changes.get(field.name, getattr(value, field.name))
            for field in value.__dataclass_fields__.values()
        }
    )


class IdentityCommandAndAtomicTests(unittest.TestCase):
    def test_gate_and_browser_scripts_are_executable(self):
        self.assertTrue(os.access(TOOL_PATH, os.X_OK))
        self.assertTrue(
            os.access(
                ROOT / "deploy" / "openwebui" / "browser-failure-smoke.cjs",
                os.X_OK,
            )
        )

    def test_content_images_urls_and_commands_are_fixed(self):
        image = "sha256:" + "a" * 64
        self.assertEqual(TOOL.normalized_content_image(image), (image, image))
        with self.assertRaisesRegex(TOOL.FailureGateError, "immutable"):
            TOOL.normalized_content_image("browser:latest")
        self.assertEqual(
            TOOL.normalized_ready_url("http://172.20.0.1:8000/readyz"),
            "http://172.20.0.1:8000/readyz",
        )
        with self.assertRaisesRegex(TOOL.FailureGateError, "differs"):
            TOOL.normalized_ready_url("http://127.0.0.1:8000/readyz")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "browser.cjs"
            token = root / "token"
            output = root / "output"
            control = root / "control"
            script.write_text("", encoding="ascii")
            token.write_text("secret-value", encoding="ascii")
            output.mkdir()
            control.mkdir()
            command = TOOL.build_browser_command(
                docker="docker",
                image=image,
                name="failure-browser",
                script=script,
                token_file=token,
                browser_output=output,
                control_dir=control,
                openwebui_url="http://127.0.0.1:3000/",
                uid=os.geteuid(),
                gid=os.getegid(),
                control_timeout_ms=180_000,
            )
        self.assertEqual(command.count("--mount"), 4)
        self.assertIn("--network=host", command)
        self.assertNotIn("secret-value", "\n".join(command))
        control_mount = next(item for item in command if "dst=/run/control" in item)
        self.assertTrue(control_mount.endswith(",readonly"))
        probe = TOOL.build_ready_probe_command(
            docker="docker",
            image=image,
            network="open-webui-network",
            ready_url="http://172.20.0.1:8000/readyz",
            timeout_seconds=30,
            uid=1000,
            gid=1000,
            name="ready-probe",
        )
        self.assertIn("--read-only", probe)
        self.assertIn("--memory=128m", probe)

    def test_service_network_and_process_identity_parsers(self):
        account = pwd.getpwuid(os.geteuid())
        service_raw = (
            f"MainPID={os.getpid()}\nUser={account.pw_name}\nActiveState=active\n"
            "SubState=running\nNRestarts=2\n"
        ).encode()
        service = TOOL.query_service_identity(
            "systemctl", "ullm-openai.service", lambda *_args, **_kwargs: service_raw
        )
        self.assertEqual(service.main_pid, os.getpid())
        process = TOOL.read_process_identity(os.getpid())
        self.assertEqual(process.pid, os.getpid())
        network = TOOL.query_docker_network(
            "docker",
            "open-webui-network",
            lambda *_args, **_kwargs: b"a" * 64 + b"|172.20.0.0/16|172.20.0.1\n",
        )
        self.assertEqual(network.gateway, "172.20.0.1")

    def test_journal_priority_and_browser_artifact_snapshot_are_fail_closed(self):
        event = lifecycle_records()[0].event
        message = TOOL.compact_json(event).decode("ascii")
        journal = {
            "__CURSOR": "cursor-one",
            "__MONOTONIC_TIMESTAMP": "100",
            "_BOOT_ID": "a" * 32,
            "_PID": "1000",
            "_SYSTEMD_UNIT": "ullm-openai.service",
            "PRIORITY": "6",
            "MESSAGE": message,
        }
        _cursor, parsed = TOOL.validate_journal_record(
            TOOL.compact_json(journal),
            service="ullm-openai.service",
            boot_id="a" * 32,
            cursors=set(),
            lifecycle_payloads=set(),
        )
        self.assertIsNotNone(parsed)
        del journal["PRIORITY"]
        with self.assertRaisesRegex(TOOL.FailureGateError, "required fields"):
            TOOL.validate_journal_record(
                TOOL.compact_json(journal),
                service="ullm-openai.service",
                boot_id="a" * 32,
                cursors=set(),
                lifecycle_payloads=set(),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / TOOL.BROWSER_CONTAINER_OUTPUT_DIR_NAME
            destination = root / "browser"
            source.mkdir(mode=0o700)
            destination.mkdir(mode=0o700)
            (source / TOOL.SCREENSHOT_NAME).write_bytes(SCREENSHOT)
            summary = b'{"summary":true}'
            (source / TOOL.BROWSER_SUMMARY_NAME).write_bytes(summary + b"\n")
            TOOL.snapshot_validated_browser_artifacts(
                source,
                destination,
                expected_summary=summary,
                expected_screenshot_sha256=SCREENSHOT_SHA,
            )
            self.assertEqual(
                (destination / TOOL.SCREENSHOT_NAME).read_bytes(), SCREENSHOT
            )
            (source / TOOL.BROWSER_SUMMARY_NAME).write_bytes(b'{"changed":true}\n')
            with self.assertRaisesRegex(TOOL.FailureGateError, "changed after"):
                TOOL.snapshot_validated_browser_artifacts(
                    source,
                    root / "other",
                    expected_summary=summary,
                    expected_screenshot_sha256=SCREENSHOT_SHA,
                )

    def test_systemd_manager_unit_records_are_accepted_but_not_lifecycle(self):
        manager = {
            "__CURSOR": "manager-cursor",
            "__MONOTONIC_TIMESTAMP": "1014113384889",
            "_BOOT_ID": "a" * 32,
            "_PID": "1",
            "_SYSTEMD_UNIT": "init.scope",
            "UNIT": "ullm-openai.service",
            "SYSLOG_IDENTIFIER": "systemd",
            "PRIORITY": "5",
            "MESSAGE": (
                "ullm-openai.service: Main process exited, "
                "code=exited, status=1/FAILURE"
            ),
        }
        cursor, lifecycle_record = TOOL.validate_journal_record(
            TOOL.compact_json(manager),
            service="ullm-openai.service",
            boot_id="a" * 32,
            cursors=set(),
            lifecycle_payloads=set(),
        )
        self.assertEqual(cursor, "manager-cursor")
        self.assertIsNone(lifecycle_record)

        manager["UNIT"] = "foreign.service"
        with self.assertRaisesRegex(TOOL.FailureGateError, "service identity"):
            TOOL.validate_journal_record(
                TOOL.compact_json(manager),
                service="ullm-openai.service",
                boot_id="a" * 32,
                cursors=set(),
                lifecycle_payloads=set(),
            )

    def test_recovery_rejects_reused_worker_pid_or_starttime(self):
        account = pwd.getpwuid(os.geteuid())
        initial_service = TOOL.ServiceIdentity(
            "ullm-openai.service",
            100,
            account.pw_name,
            account.pw_uid,
            account.pw_gid,
            2,
        )
        recovered_service = dataclass_replace(initial_service, main_pid=101, restarts=3)
        initial_worker = TOOL.ProcessIdentity(200, 100, 10, account.pw_uid, "a" * 64)
        reused_pid = TOOL.ProcessIdentity(200, 101, 20, account.pw_uid, "a" * 64)
        with (
            mock.patch.object(
                TOOL, "query_service_identity", return_value=recovered_service
            ),
            mock.patch.object(TOOL, "query_worker_identity", return_value=reused_pid),
        ):
            with self.assertRaisesRegex(TOOL.FailureGateError, "worker identity"):
                TOOL.wait_recovered_service(
                    systemctl="systemctl",
                    service="ullm-openai.service",
                    initial_service=initial_service,
                    initial_worker=initial_worker,
                    deadline_ns=TOOL.time.monotonic_ns() + 1_000_000_000,
                )

    def test_live_pidfd_pin_does_not_signal_process(self):
        process = TOOL.read_process_identity(os.getpid())
        descriptor = TOOL.open_worker_pidfd(process)
        os.close(descriptor)

    def test_secret_atomic_and_failure_output_are_redacted(self):
        guard = TOOL.SecretGuard([b"very-secret-token"])
        with self.assertRaisesRegex(TOOL.FailureGateError, "forbidden"):
            guard.reject(b"very-secret-token", "fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "control"
            created, control_sha = TOOL.create_control_file(
                path, "worker_killed", "f" * 64
            )
            self.assertGreater(created, 0)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            self.assertEqual(control_sha, digest(path.read_bytes()))
            failed = TOOL.AtomicRunDirectory(root / "failed")
            failed.abort()
            self.assertFalse((root / "failed").exists())
        stderr = io.StringIO()
        with (
            mock.patch.object(
                TOOL, "execute", side_effect=RuntimeError("very-secret-token")
            ),
            mock.patch.object(TOOL, "parse_args", return_value=object()),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(TOOL.main([]), 1)
        self.assertEqual(stderr.getvalue(), "OpenWebUI failure gate failed\n")


if __name__ == "__main__":
    unittest.main()
