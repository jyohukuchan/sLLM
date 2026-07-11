from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import pwd
import signal
import socket
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run-openwebui-stop-gate.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("run_openwebui_stop_gate", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lifecycle(name: str, timestamp: int, request: str, completion: str, **fields):
    return {
        "schema_version": TOOL.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": timestamp,
        "request_id": request,
        "completion_id": completion,
        **fields,
    }


def lifecycle_trace(request: str, completion: str, start: int, *, cancelled: bool):
    values = [
        lifecycle(
            "request_admitted",
            start,
            request,
            completion,
            stream=True,
            prompt_tokens=32,
            max_completion_tokens=64,
        ),
        lifecycle(
            "request_started",
            start + 1,
            request,
            completion,
            stream=True,
            prompt_tokens=32,
            admit_to_start_ns=1,
        ),
        lifecycle(
            "request_progress",
            start + 2,
            request,
            completion,
            phase="prefill",
            processed_prompt_tokens=32,
            prompt_tokens=32,
        ),
        lifecycle(
            "request_first_token",
            start + 3,
            request,
            completion,
            stream=True,
            completion_tokens=1,
        ),
    ]
    if cancelled:
        values.append(
            lifecycle(
                "request_cancel_requested",
                start + 4,
                request,
                completion,
                stream=True,
                reason="client_disconnect",
                admit_to_cancel_ns=4,
            )
        )
    values.append(
        lifecycle(
            "request_released",
            start + 5,
            request,
            completion,
            stream=True,
            outcome="cancelled" if cancelled else "stop",
            cancel_reason="client_disconnect" if cancelled else None,
            prompt_tokens=32,
            completion_tokens=1,
            reset_complete=True,
            admit_to_start_ns=1,
            start_to_release_ns=4,
            admit_to_release_ns=5,
        )
    )
    return values


def action(index: int, name: str):
    input_sha = None
    if index == 0:
        input_sha = "a" * 64
    elif index == 1:
        input_sha = digest(TOOL.MODEL_ID)
    elif index == 2:
        input_sha = digest(TOOL.STOP_PROMPT)
    elif index == 6:
        input_sha = digest(TOOL.RECOVERY_PROMPT)
    screenshot = name == "click_stop"
    enabled = True if name in {"submit_chat", "click_stop", "wait_ready"} else None
    return {
        "browser_case": TOOL.BROWSER_CASE,
        "action_index": index,
        "action": name,
        "selector": None,
        "input_sha256": input_sha,
        "started_monotonic_ns": str(100 + index * 10),
        "completed_monotonic_ns": str(105 + index * 10),
        "result": {
            "visible": True,
            "enabled": enabled,
            "text_utf8_bytes": 10
            if name in {"wait_visible", "click_stop", "wait_ready"}
            else None,
            "text_sha256": "b" * 64
            if name in {"wait_visible", "click_stop", "wait_ready"}
            else None,
        },
        "screenshot_file": "browser/openwebui-stop-before.png" if screenshot else None,
        "screenshot_sha256": "c" * 64 if screenshot else None,
    }


def socket_event(
    sequence: int,
    target: str,
    kind: str,
    *,
    observed: int,
    done=False,
    content=False,
):
    return {
        "sequence": sequence,
        "observed_monotonic_ns": str(observed),
        "correlation_target": target,
        "type": kind,
        "done": done,
        "has_error": False,
        "content_utf8_bytes": 8 if content else 0,
        "content_sha256": "d" * 64 if content else None,
    }


def identity(prefix: str, value: str):
    return {
        f"{prefix}_utf8_bytes": len(value.encode()),
        f"{prefix}_sha256": digest(value),
    }


def interim_fixture():
    nonce = "e" * 64
    content = f"{TOOL.CONTROL_SCHEMA}:{nonce}\n".encode()
    return {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_stop_gateway_release_wait",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "2000",
        "browser_actions": [
            action(index, name) for index, name in enumerate(TOOL.FINAL_ACTIONS[:6])
        ],
        "socket_correlation": {
            "target": {
                **identity("chat_id", "chat-one"),
                **identity("message_id", "message-one"),
            },
            "click_completed_monotonic_ns": "145",
            "cancel_first_observed_monotonic_ns": "150",
            "cancel_event_count": 1,
            "done_after_click_count": 0,
            "content_after_cancel_count": 0,
        },
        "socket_events": [
            socket_event(
                0, "cancel_target", "chat:completion", observed=130, content=True
            ),
            socket_event(1, "cancel_target", "chat:tasks:cancel", observed=150),
        ],
        "page_error_count": 0,
        "gateway_release_control": {
            "control_schema": TOOL.CONTROL_SCHEMA,
            "control_file": TOOL.CONTROL_CONTAINER_PATH,
            "nonce": nonce,
            "content_utf8_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "timeout_ms": 30_000,
        },
    }


def final_fixture(interim, screenshot: bytes):
    actions = copy.deepcopy(interim["browser_actions"])
    actions.extend(
        action(index, name)
        for index, name in enumerate(TOOL.FINAL_ACTIONS[6:], start=6)
    )
    nonce = interim["gateway_release_control"]["nonce"]
    content = f"{TOOL.CONTROL_SCHEMA}:{nonce}\n".encode()
    events = copy.deepcopy(interim["socket_events"])
    events.extend(
        (
            socket_event(
                2, "recovery_target", "chat:completion", observed=170, content=True
            ),
            socket_event(
                3, "recovery_target", "chat:completion", observed=178, done=True
            ),
        )
    )
    return {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_stop_smoke",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "3000",
        "browser_actions": actions,
        "socket_correlation": {
            "target": copy.deepcopy(interim["socket_correlation"]["target"]),
            "click_started_monotonic_ns": "140",
            "click_completed_monotonic_ns": "145",
            "cancel_first_observed_monotonic_ns": "150",
            "cancel_event_count": 1,
            "done_after_click_count": 0,
            "content_after_cancel_count": 0,
            "recovery": {
                **identity("chat_id", "chat-one"),
                **identity("message_id", "message-two"),
                "submit_completed_monotonic_ns": "165",
                "done_observed_monotonic_ns": "178",
                "done_event_count": 1,
                "cancel_event_count": 0,
            },
        },
        "page_error_count": 0,
        "page_errors": [],
        "socket_events": events,
        "gateway_release_control": {
            "control_schema": TOOL.CONTROL_SCHEMA,
            **identity("control_file", TOOL.CONTROL_CONTAINER_PATH),
            "nonce_sha256": digest(nonce),
            "content_utf8_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "requested_monotonic_ns": "2000",
            "observed_monotonic_ns": "2001",
        },
        "screenshot": {
            "screenshot_file": "browser/openwebui-stop-before.png",
            "screenshot_bytes": len(screenshot),
            "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
        },
    }


class JsonAndLifecycleTests(unittest.TestCase):
    def test_strict_json_and_lifecycle_reject_mutation(self):
        with self.assertRaisesRegex(TOOL.StopGateError, "duplicate"):
            TOOL.strict_json_object(b'{"a":1,"a":2}', "fixture")
        with self.assertRaisesRegex(TOOL.StopGateError, "UTF-8"):
            TOOL.strict_json_object(b'{"a":"\xff"}', "fixture")
        event = lifecycle_trace("req-a", "chatcmpl-a", 10, cancelled=True)[0]
        raw = TOOL.compact_json(event)
        self.assertEqual(TOOL.validate_lifecycle_payload(raw), event)
        with self.assertRaisesRegex(TOOL.StopGateError, "canonical"):
            TOOL.validate_lifecycle_payload(b" " + raw)

    def test_gateway_machine_enforces_stop_then_recovery_without_overlap(self):
        machine = TOOL.LifecycleMachine()
        for event in lifecycle_trace("req-a", "chatcmpl-a", 100, cancelled=True):
            machine.consume(event)
        TOOL.validate_gateway_traces(
            machine,
            click_completed_ns=100,
            control_created_ns=106,
            final=False,
        )
        for event in lifecycle_trace("req-b", "chatcmpl-b", 200, cancelled=False):
            machine.consume(event)
        TOOL.validate_gateway_traces(
            machine,
            click_completed_ns=100,
            control_created_ns=106,
            final=True,
        )
        self.assertEqual(machine.max_active, 1)

        overlapping = TOOL.LifecycleMachine()
        overlapping.consume(
            lifecycle_trace("req-a", "chatcmpl-a", 1, cancelled=True)[0]
        )
        with self.assertRaisesRegex(TOOL.StopGateError, "overlapping"):
            overlapping.consume(
                lifecycle_trace("req-b", "chatcmpl-b", 2, cancelled=False)[0]
            )

    def test_observer_uses_kernel_credentials_and_rejects_wrong_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            writer = TOOL.AtomicLineWriter(
                root / "observer.jsonl", maximum_bytes=1024 * 1024
            )
            observer = TOOL.LifecycleObserver(
                root / "observer.sock", os.getpid(), os.geteuid(), writer
            )
            observer.open()
            event = lifecycle_trace(
                "req-a", "chatcmpl-a", time.monotonic_ns(), cancelled=True
            )[0]
            payload = TOOL.compact_json(event)
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(payload, os.fspath(root / "observer.sock"))
            observer.wait_for(
                lambda machine: len(machine.traces) == 1,
                time.monotonic_ns() + 1_000_000_000,
            )
            record = observer.snapshot()[0]
            self.assertEqual(
                (record.sender_pid, record.sender_uid), (os.getpid(), os.geteuid())
            )
            observer.close()
            writer.commit()
            self.assertEqual((root / "observer.jsonl").read_bytes(), payload + b"\n")

            wrong_writer = TOOL.AtomicLineWriter(
                root / "wrong.jsonl", maximum_bytes=1024 * 1024
            )
            wrong = TOOL.LifecycleObserver(
                root / "wrong.sock", os.getpid() + 1, os.geteuid(), wrong_writer
            )
            wrong.open()
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(payload, os.fspath(root / "wrong.sock"))
            with self.assertRaisesRegex(TOOL.StopGateError, "sender PID"):
                wrong.wait_for(
                    lambda _machine: False, time.monotonic_ns() + 1_000_000_000
                )
            with self.assertRaises(TOOL.StopGateError):
                wrong.close()
            wrong_writer.abort()

    def test_observer_close_drains_datagrams_already_queued(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            writer = TOOL.AtomicLineWriter(
                root / "drain.jsonl", maximum_bytes=1024 * 1024
            )
            observer = TOOL.LifecycleObserver(
                root / "drain.sock", os.getpid(), os.geteuid(), writer
            )
            observer.open()
            events = lifecycle_trace(
                "req-drain",
                "chatcmpl-drain",
                time.monotonic_ns(),
                cancelled=True,
            )[:2]
            payloads = [TOOL.compact_json(event) for event in events]
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                for payload in payloads:
                    sender.sendto(payload, os.fspath(root / "drain.sock"))
            observer.close()
            writer.commit()
            self.assertEqual(
                (root / "drain.jsonl").read_bytes(), b"\n".join(payloads) + b"\n"
            )


class JournalAndControlTests(unittest.TestCase):
    def test_journal_message_matches_payload_and_cursor_is_unique(self):
        event = lifecycle_trace("req-a", "chatcmpl-a", 10, cancelled=True)[0]
        payload = TOOL.compact_json(event)
        record = {
            "__CURSOR": "cursor-1",
            "__MONOTONIC_TIMESTAMP": "10",
            "_BOOT_ID": "a" * 32,
            "_PID": "123",
            "_SYSTEMD_UNIT": "ullm-openai.service",
            "MESSAGE": "INFO:     " + payload.decode(),
        }
        cursors: set[str] = set()
        lifecycle_payloads: set[bytes] = set()
        cursor, observed = TOOL.validate_journal_record(
            TOOL.compact_json(record),
            service="ullm-openai.service",
            main_pid=123,
            boot_id="a" * 32,
            cursors=cursors,
            lifecycle_payloads=lifecycle_payloads,
        )
        self.assertEqual((cursor, observed), ("cursor-1", payload))
        cursors.add(cursor)
        lifecycle_payloads.add(payload)
        with self.assertRaisesRegex(TOOL.StopGateError, "cursor is duplicated"):
            TOOL.validate_journal_record(
                TOOL.compact_json(record),
                service="ullm-openai.service",
                main_pid=123,
                boot_id="a" * 32,
                cursors=cursors,
                lifecycle_payloads=lifecycle_payloads,
            )
        record["__CURSOR"] = "cursor-2"
        record["MESSAGE"] = "INFO:     " + payload.decode().replace("req-a", "req-x")
        _cursor, altered = TOOL.validate_journal_record(
            TOOL.compact_json(record),
            service="ullm-openai.service",
            main_pid=123,
            boot_id="a" * 32,
            cursors=cursors,
            lifecycle_payloads=lifecycle_payloads,
        )
        self.assertIsNotNone(altered)
        with self.assertRaisesRegex(TOOL.StopGateError, "payload bytes differ"):
            TOOL.require_correlated_prefix([payload], [altered])

    def test_control_is_exclusive_regular_0600_with_exact_nonce(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gateway-released"
            nonce = "f" * 64
            _created, content_sha = TOOL.create_control_file(path, nonce)
            expected = f"{TOOL.CONTROL_SCHEMA}:{nonce}\n".encode()
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(content_sha, hashlib.sha256(expected).hexdigest())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(TOOL.StopGateError, "exclusive"):
                TOOL.create_control_file(path, nonce)


class BrowserContractTests(unittest.TestCase):
    def test_socket_state_events_are_allowed_only_without_content_or_done(self):
        events = [
            socket_event(0, "cancel_target", "chat:active", observed=120),
            socket_event(1, "cancel_target", "chat:outlet", observed=125),
            socket_event(
                2, "cancel_target", "chat:completion", observed=130, content=True
            ),
            socket_event(3, "cancel_target", "chat:tasks:cancel", observed=150),
        ]
        result = TOOL.validate_socket_events(events, final=False)
        self.assertEqual(result["target_cancel_count"], 1)

        changed = copy.deepcopy(events)
        changed[0]["content_utf8_bytes"] = 1
        changed[0]["content_sha256"] = "d" * 64
        with self.assertRaisesRegex(TOOL.StopGateError, "state event"):
            TOOL.validate_socket_events(changed, final=False)

        changed = copy.deepcopy(events)
        changed[1]["done"] = True
        with self.assertRaisesRegex(TOOL.StopGateError, "state event"):
            TOOL.validate_socket_events(changed, final=False)

    def test_interim_control_and_final_summary_are_cross_checked(self):
        screenshot = b"fake-png-bytes"
        interim = interim_fixture()
        interim["browser_actions"][4]["screenshot_sha256"] = hashlib.sha256(
            screenshot
        ).hexdigest()
        guard = TOOL.SecretGuard([b"test-secret", TOOL.STOP_PROMPT.encode()])
        nonce = TOOL.validate_interim(interim, guard)
        self.assertEqual(nonce, "e" * 64)
        final = final_fixture(interim, screenshot)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screenshot_path = root / TOOL.SCREENSHOT_NAME
            summary_path = root / TOOL.BROWSER_SUMMARY_NAME
            screenshot_path.write_bytes(screenshot)
            raw = json.dumps(final, separators=(",", ":")).encode()
            summary_path.write_bytes(raw + b"\n")
            result = TOOL.validate_final_browser(
                final,
                interim,
                raw,
                summary_path,
                screenshot_path,
                guard,
            )
            self.assertEqual(result["action_count"], 9)
            self.assertEqual(
                result["screenshot_sha256"], hashlib.sha256(screenshot).hexdigest()
            )

            changed = copy.deepcopy(final)
            changed["socket_correlation"]["recovery"]["chat_id_sha256"] = "0" * 64
            with self.assertRaisesRegex(TOOL.StopGateError, "same-chat"):
                TOOL.validate_final_browser(
                    changed,
                    interim,
                    raw,
                    summary_path,
                    screenshot_path,
                    guard,
                )

    def test_interim_rejects_nonce_and_action_mutation(self):
        interim = interim_fixture()
        interim["gateway_release_control"]["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(TOOL.StopGateError, "control content"):
            TOOL.validate_interim(interim, TOOL.SecretGuard([]))
        interim = interim_fixture()
        interim["browser_actions"][2]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(TOOL.StopGateError, "Stop prompt"):
            TOOL.validate_interim(interim, TOOL.SecretGuard([]))
        with self.assertRaisesRegex(TOOL.StopGateError, "control content"):
            TOOL.validate_interim(
                interim_fixture(),
                TOOL.SecretGuard([]),
                expected_timeout_ms=29_999,
            )

    def test_docker_command_has_only_four_host_binds_and_no_token_value(self):
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
                image="browser@sha256:" + "a" * 64,
                name="gate-container",
                script=script,
                token_file=token,
                browser_output=output,
                control_dir=control,
                openwebui_url="http://127.0.0.1:3000",
                uid=os.geteuid(),
                gid=os.getegid(),
                gateway_wait_ms=30_000,
            )
            self.assertIn("--network=host", command)
            self.assertEqual(command.count("--mount"), 4)
            self.assertNotIn("secret-value", "\n".join(command))
            self.assertIn("node", command)
            self.assertIn(TOOL.BROWSER_SCRIPT_CONTAINER_PATH, command)
            self.assertEqual(
                TOOL.normalized_browser_image("sha256:" + "b" * 64),
                ("sha256:" + "b" * 64, "sha256:" + "b" * 64),
            )
            with self.assertRaisesRegex(TOOL.StopGateError, "immutable"):
                TOOL.normalized_browser_image("browser:latest")


class IdentitySecretAndAtomicTests(unittest.TestCase):
    def test_service_identity_requires_active_main_pid_and_resolves_user(self):
        account = pwd.getpwuid(os.geteuid())
        raw = (
            f"MainPID=123\nUser={account.pw_name}\nActiveState=active\n"
            "SubState=running\nNRestarts=2\n"
        ).encode()
        identity = TOOL.query_service_identity(
            "systemctl", "ullm-openai.service", lambda *_args, **_kwargs: raw
        )
        self.assertEqual(
            (identity.main_pid, identity.uid, identity.restarts), (123, os.geteuid(), 2)
        )
        with self.assertRaisesRegex(TOOL.StopGateError, "not active"):
            TOOL.query_service_identity(
                "systemctl",
                "ullm-openai.service",
                lambda *_args, **_kwargs: raw.replace(
                    b"ActiveState=active", b"ActiveState=failed"
                ),
            )

    def test_secret_guard_and_failure_output_never_echo_secret(self):
        guard = TOOL.SecretGuard([b"very-secret-token"])
        with self.assertRaisesRegex(TOOL.StopGateError, "forbidden"):
            guard.reject(b'{"value":"very-secret-token"}', "summary")
        stderr = io.StringIO()
        with (
            mock.patch.object(
                TOOL, "execute", side_effect=TOOL.StopGateError("very-secret-token")
            ),
            mock.patch.object(TOOL, "parse_args", return_value=object()),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(TOOL.main([]), 1)
        self.assertEqual(stderr.getvalue(), "OpenWebUI Stop gate failed\n")

    def test_atomic_directory_abort_never_leaves_pass_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed_path = root / "failed"
            failed = TOOL.AtomicRunDirectory(failed_path)
            (failed.stage / "summary.json").write_text("partial", encoding="ascii")
            failed.abort()
            self.assertFalse(failed_path.exists())
            self.assertFalse(failed.stage.exists())

            passed_path = root / "passed"
            passed = TOOL.AtomicRunDirectory(passed_path)
            (passed.stage / "summary.json").write_text("complete", encoding="ascii")
            passed.publish()
            self.assertEqual((passed_path / "summary.json").read_text(), "complete")

    def test_private_snapshot_is_exact_read_only_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot"
            TOOL.write_private_snapshot(path, b"fixed-input", "test input")
            self.assertEqual(path.read_bytes(), b"fixed-input")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            with self.assertRaisesRegex(TOOL.StopGateError, "snapshot"):
                TOOL.write_private_snapshot(path, b"replacement", "test input")

    def test_process_group_cleanup_kills_descendants_after_leader_exit(self):
        process = mock.Mock()
        process.pid = 4242
        process.wait.return_value = 0
        existence = iter([True, True, False])
        with (
            mock.patch.object(
                TOOL,
                "process_group_exists",
                side_effect=lambda _group: next(existence, False),
            ),
            mock.patch.object(TOOL.os, "killpg") as killpg,
            mock.patch.object(TOOL.time, "sleep"),
        ):
            TOOL.terminate_process_group(process)
        self.assertEqual(killpg.call_args_list, [mock.call(4242, signal.SIGTERM)])


if __name__ == "__main__":
    unittest.main()
