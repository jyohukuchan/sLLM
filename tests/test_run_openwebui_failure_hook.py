from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run-openwebui-failure-hook.py"
COLLECTOR_PATH = ROOT / "tools" / "collect-sq8-openwebui-release.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_module("run_openwebui_failure_hook", TOOL_PATH)
COLLECTOR = load_module("collect_sq8_openwebui_release_for_hook", COLLECTOR_PATH)


def digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


SCREENSHOT = b"\x89PNG\r\n\x1a\nformal-failure-hook"
SCREENSHOT_SHA = digest(SCREENSHOT)
ACTION_STARTS = (100, 120, 140, 160, 200, 280, 300, 320, 340)
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
WORKER_NONCE = "d" * 64
RECOVERY_NONCE = "e" * 64
TEST_BINDINGS = TOOL.BundleBindings(
    gate_source_sha256=digest("gate-source"),
    browser_script_sha256=digest("browser-script"),
    browser_image_reference_sha256=digest("browser-reference"),
    probe_image_reference_sha256=digest("probe-reference"),
    service_unit_sha256=digest("ullm-openai.service"),
)


def action(index: int) -> dict:
    name = TOOL.EXPECTED_ACTIONS[index]
    input_sha = digest(f"redacted-input-{index}") if index in {0, 1, 2, 6} else None
    carries_text = name in {"wait_visible", "wait_failed"} or index == 8
    return {
        "browser_case": TOOL.BROWSER_CASE,
        "action_index": index,
        "action": name,
        "selector": TOOL.EXPECTED_SELECTORS[index],
        "input_sha256": input_sha,
        "started_monotonic_ns": str(ACTION_STARTS[index]),
        "completed_monotonic_ns": str(ACTION_STARTS[index] + 5),
        "result": {
            "visible": True,
            "enabled": True if name in {"submit_chat", "wait_ready"} else None,
            "text_utf8_bytes": 12 if carries_text else None,
            "text_sha256": digest(f"redacted-text-{index}") if carries_text else None,
        },
        "screenshot_file": "browser/post-header-failure.png" if index == 4 else None,
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
        "content_sha256": digest(f"content-{sequence}") if content else None,
    }


def control_raw(stage: str, nonce: str) -> bytes:
    return f"ullm.openwebui.failure_control.v1:{stage}:{nonce}\n".encode("ascii")


def clear_control(stage: str, nonce: str) -> dict:
    path = {
        "worker_killed": "/run/control/worker-killed",
        "gateway_recovered": "/run/control/gateway-recovered",
    }[stage]
    raw = control_raw(stage, nonce)
    return {
        "control_schema": "ullm.openwebui.failure_control.v1",
        "control_stage": stage,
        "control_file": path,
        "nonce": nonce,
        "content_utf8_bytes": len(raw),
        "content_sha256": digest(raw),
        "timeout_ms": 180_000,
    }


def redacted_control(stage: str, nonce: str, requested: int, observed: int) -> dict:
    path = {
        "worker_killed": "/run/control/worker-killed",
        "gateway_recovered": "/run/control/gateway-recovered",
    }[stage]
    raw = control_raw(stage, nonce)
    return {
        "control_schema": "ullm.openwebui.failure_control.v1",
        "control_stage": stage,
        "control_file_utf8_bytes": len(path.encode("utf-8")),
        "control_file_sha256": digest(path),
        "nonce_sha256": digest(nonce),
        "content_utf8_bytes": len(raw),
        "content_sha256": digest(raw),
        "requested_monotonic_ns": str(requested),
        "observed_monotonic_ns": str(observed),
    }


def browser_fixture() -> tuple[dict, dict, dict]:
    actions = [action(index) for index in range(9)]
    events = [
        socket_event(0, "failure_target", "chat:completion", 150, content=True),
        socket_event(1, "failure_target", "chat:completion", 180, error=True),
        socket_event(2, "failure_target", "chat:tasks:cancel", 190),
        socket_event(3, "recovery_target", "chat:completion", 330, content=True),
        socket_event(4, "recovery_target", "chat:completion", 335, done=True),
    ]
    worker_control = redacted_control("worker_killed", WORKER_NONCE, 170, 175)
    recovery_control = redacted_control("gateway_recovered", RECOVERY_NONCE, 240, 270)
    first = {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_worker_kill_wait",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "169",
        "browser_actions": copy.deepcopy(actions[:4]),
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "submit_completed_monotonic_ns": "145",
            "visible_completed_monotonic_ns": "165",
            "pre_fault_done_count": 0,
            "pre_fault_error_count": 0,
            "pre_fault_cancel_count": 0,
        },
        "socket_events": copy.deepcopy(events[:1]),
        "page_error_count": 0,
        "worker_killed_control": clear_control("worker_killed", WORKER_NONCE),
    }
    second = {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_gateway_recovery_wait",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "240",
        "browser_actions": copy.deepcopy(actions[:5]),
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "error_first_observed_monotonic_ns": "180",
            "cancel_first_observed_monotonic_ns": "190",
            "error_event_count": 1,
            "cancel_event_count": 1,
            "done_after_fault_count": 0,
            "content_after_error_count": 0,
        },
        "socket_events": copy.deepcopy(events[:3]),
        "page_error_count": 0,
        "worker_killed_control": copy.deepcopy(worker_control),
        "gateway_recovered_control": clear_control("gateway_recovered", RECOVERY_NONCE),
    }
    final = {
        "schema_version": TOOL.BROWSER_SCHEMA,
        "record_type": "openwebui_failure_smoke",
        "browser_case": TOOL.BROWSER_CASE,
        "observed_monotonic_ns": "360",
        "browser_actions": actions,
        "socket_correlation": {
            "target": copy.deepcopy(TARGET),
            "error_first_observed_monotonic_ns": "180",
            "cancel_first_observed_monotonic_ns": "190",
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
        "socket_events": events,
        "controls": {
            "worker_killed": worker_control,
            "gateway_recovered": recovery_control,
        },
        "screenshot": {
            "screenshot_file": "browser/post-header-failure.png",
            "screenshot_bytes": len(SCREENSHOT),
            "screenshot_sha256": SCREENSHOT_SHA,
        },
    }
    return first, second, final


def write_file(path: Path, raw: bytes, mode: int) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


def build_bundle(path: Path, bindings=TEST_BINDINGS) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    browser = path / "browser"
    browser.mkdir(mode=0o700)
    browser.chmod(0o700)
    first, second, final = browser_fixture()
    final_raw = (
        json.dumps(final, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    stdout_raw = (
        b"".join(
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
            for value in (first, second)
        )
        + final_raw
    )
    write_file(browser / "browser-stdout.jsonl", stdout_raw, 0o600)
    write_file(browser / "openwebui-failure-summary.json", final_raw, 0o400)
    write_file(browser / "post-header-failure.png", SCREENSHOT, 0o400)

    fault = {
        "schema_version": TOOL.GATE_SCHEMA,
        "record_type": "fault_injection",
        "injection": "post_header_worker_kill",
        "target_pid": 1002,
        "target_starttime_ticks": 9002,
        "target_parent_pid": 1001,
        "signal": "SIGKILL",
        "command": "signal.pidfd_send_signal",
        "started_monotonic_ns": 170,
        "completed_monotonic_ns": 171,
    }
    fault_raw = canonical(fault)
    write_file(path / "fault-injection.json", fault_raw, 0o600)
    readiness = {
        "schema_version": TOOL.GATE_SCHEMA,
        "record_type": "readiness_evidence",
        "network_id": "network-identity-redacted",
        "subnet": "172.20.0.0/16",
        "gateway": "172.20.0.1",
        "initial": {
            "started_monotonic_ns": 10,
            "completed_monotonic_ns": 20,
            "status": 200,
        },
        "recovered": {
            "started_monotonic_ns": 250,
            "completed_monotonic_ns": 260,
            "status": 200,
        },
    }
    readiness_raw = canonical(readiness)
    write_file(path / "readiness-evidence.json", readiness_raw, 0o600)
    journal_raw = b'{"__CURSOR":"cursor-one"}\n{"__CURSOR":"cursor-two"}\n'
    write_file(path / "service-journal.raw.jsonl", journal_raw, 0o600)
    summary = {
        "schema_version": TOOL.GATE_SCHEMA,
        "passed": True,
        "service": {
            "unit_sha256": bindings.service_unit_sha256,
            "initial_gateway_pid": 1001,
            "recovered_gateway_pid": 2001,
            "initial_worker_pid": 1002,
            "recovered_worker_pid": 2002,
            "initial_worker_starttime_ticks": 9002,
            "recovered_worker_starttime_ticks": 9902,
            "initial_restart_count": 3,
            "recovered_restart_count": 4,
            "restart_delta": 1,
            "boot_id_sha256": digest("boot"),
        },
        "browser": {
            "image_reference_sha256": bindings.browser_image_reference_sha256,
            "image_content_digest": f"sha256:{digest('browser-image')}",
            "script_sha256": bindings.browser_script_sha256,
            "action_count": 9,
            "socket_event_count": 5,
            "screenshot_sha256": SCREENSHOT_SHA,
            "stdout_lines": 3,
            "stdout_bytes": len(stdout_raw),
            "stdout_sha256": digest(stdout_raw),
            "stderr_bytes": 0,
            "stderr_sha256": TOOL.EMPTY_SHA256,
        },
        "fault": {
            "target_request_sha256": digest("target-request"),
            "target_completion_sha256": digest("target-completion"),
            "worker_fatal_monotonic_ns": 180,
            "signal_to_fatal_ns": 10,
            "fault_artifact_sha256": digest(fault_raw),
            "kill_control_sha256": digest(control_raw("worker_killed", WORKER_NONCE)),
        },
        "recovery": {
            "request_sha256": digest("recovery-request"),
            "completion_sha256": digest("recovery-completion"),
            "admitted_monotonic_ns": 300,
            "released_monotonic_ns": 350,
            "outcome": "stop",
            "reset_complete": True,
            "readiness_artifact_sha256": digest(readiness_raw),
            "recovery_control_sha256": digest(
                control_raw("gateway_recovered", RECOVERY_NONCE)
            ),
        },
        "gateway_journal": {
            "lifecycle_count": 1,
            "record_count": 2,
            "cursor_count": 2,
            "raw_sha256": digest(journal_raw),
            "stderr_bytes": 0,
            "stderr_sha256": TOOL.EMPTY_SHA256,
        },
        "probe": {
            "image_reference_sha256": bindings.probe_image_reference_sha256,
            "image_content_digest": f"sha256:{digest('probe-image')}",
            "network_id_sha256": digest(readiness["network_id"]),
        },
        "gate_source_sha256": bindings.gate_source_sha256,
    }
    write_file(path / "summary.json", canonical(summary), 0o600)
    return path


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def rewrite_summary(path: Path, mutate) -> None:
    summary = load_json(path / "summary.json")
    mutate(summary)
    write_file(path / "summary.json", canonical(summary), 0o600)


def rewrite_browser(path: Path, mutate) -> None:
    browser_summary_path = path / "browser" / "openwebui-failure-summary.json"
    final = load_json(browser_summary_path)
    mutate(final)
    final_raw = (
        json.dumps(final, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    stdout_path = path / "browser" / "browser-stdout.jsonl"
    lines = stdout_path.read_bytes().splitlines(keepends=True)
    stdout_raw = b"".join(lines[:2]) + final_raw
    write_file(browser_summary_path, final_raw, 0o400)
    write_file(stdout_path, stdout_raw, 0o600)

    def update(summary: dict) -> None:
        summary["browser"]["stdout_bytes"] = len(stdout_raw)
        summary["browser"]["stdout_sha256"] = digest(stdout_raw)

    rewrite_summary(path, update)


def rewrite_fault(path: Path, mutate) -> None:
    fault_path = path / "fault-injection.json"
    fault = load_json(fault_path)
    mutate(fault)
    raw = canonical(fault)
    write_file(fault_path, raw, 0o600)
    rewrite_summary(
        path,
        lambda summary: summary["fault"].__setitem__(
            "fault_artifact_sha256", digest(raw)
        ),
    )


def args_for(bundle: Path) -> argparse.Namespace:
    return argparse.Namespace(
        failure_gate=TOOL_PATH,
        failure_bundle=bundle,
        token_file=Path("/private/release-token"),
        browser_script=ROOT / "deploy" / "openwebui" / "browser-failure-smoke.cjs",
        browser_image="sha256:" + "a" * 64,
        probe_image="sha256:" + "b" * 64,
        openwebui_url="http://192.168.0.66:3000/",
        ready_url="http://172.20.0.1:8000/readyz",
        network="open-webui-network",
        service="ullm-openai.service",
        docker="docker",
        systemctl="systemctl",
        journalctl="journalctl",
        timeout_seconds=360,
        control_timeout_ms=180_000,
        recovery_probe_timeout_seconds=180,
    )


def bindings_for_args(args: argparse.Namespace):
    return TOOL.BundleBindings(
        gate_source_sha256=digest(args.failure_gate.read_bytes()),
        browser_script_sha256=digest(args.browser_script.read_bytes()),
        browser_image_reference_sha256=digest(args.browser_image),
        probe_image_reference_sha256=digest(args.probe_image),
        service_unit_sha256=digest(args.service),
    )


class FailureHookBundleTests(unittest.TestCase):
    def test_valid_bundle_emits_collector_compatible_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_bundle(Path(directory) / "bundle")
            records = TOOL.validate_bundle(bundle, TEST_BINDINGS)
        self.assertEqual(len(records), 10)
        self.assertEqual(
            [record["record"]["record_type"] for record in records],
            ["browser_action"] * 4 + ["fault_injection"] + ["browser_action"] * 5,
        )
        self.assertEqual(
            {record["record"]["phase"] for record in records}, {TOOL.PHASE}
        )
        self.assertEqual(
            {record["record"]["case_id"] for record in records}, {TOOL.BROWSER_CASE}
        )
        browser_records = [
            record
            for record in records
            if record["record"]["record_type"] == "browser_action"
        ]
        for index, record in enumerate(browser_records):
            fields = record["record"]["fields"]
            self.assertIs(type(fields["started_monotonic_ns"]), int)
            self.assertIs(type(fields["completed_monotonic_ns"]), int)
            self.assertEqual(fields["action_index"], index)
            COLLECTOR.validate_hook_fields("browser_action", fields)
        COLLECTOR.validate_hook_fields(
            "fault_injection", records[4]["record"]["fields"]
        )
        self.assertLessEqual(
            records[3]["record"]["fields"]["completed_monotonic_ns"],
            records[4]["record"]["fields"]["started_monotonic_ns"],
        )
        self.assertLessEqual(
            records[4]["record"]["fields"]["completed_monotonic_ns"],
            records[5]["record"]["fields"]["started_monotonic_ns"],
        )

    def test_hook_output_contains_no_raw_gate_inputs_or_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = TOOL.validate_bundle(
                build_bundle(Path(directory) / "bundle"), TEST_BINDINGS
            )
        raw = b"".join(TOOL.compact_json(record) + b"\n" for record in records)
        for forbidden in (
            b"/private/release-token",
            b"http://192.168.0.66:3000/",
            b"target-request",
            b"target-completion",
            b"chat-one",
            b"message-one",
            b"redacted-input",
        ):
            self.assertNotIn(forbidden, raw)

    def test_rejects_bundle_layout_and_mode_mutations(self) -> None:
        mutations = {
            "extra root file": lambda path: write_file(path / "extra", b"x", 0o600),
            "extra browser file": lambda path: write_file(
                path / "browser" / "extra", b"x", 0o600
            ),
            "root mode": lambda path: path.chmod(0o755),
            "summary mode": lambda path: (path / "summary.json").chmod(0o644),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                bundle = build_bundle(Path(directory) / "bundle")
                mutate(bundle)
                with self.assertRaises(TOOL.FailureHookError):
                    TOOL.validate_bundle(bundle, TEST_BINDINGS)

    def test_rejects_summary_mutations(self) -> None:
        mutations = {
            "passed false": lambda value: value.__setitem__("passed", False),
            "boolean action count": lambda value: value["browser"].__setitem__(
                "action_count", True
            ),
            "restart delta": lambda value: value["service"].__setitem__(
                "restart_delta", 2
            ),
            "journal hash": lambda value: value["gateway_journal"].__setitem__(
                "raw_sha256", "0" * 64
            ),
            "extra field": lambda value: value.__setitem__("unexpected", 1),
            "gate source binding": lambda value: value.__setitem__(
                "gate_source_sha256", "0" * 64
            ),
            "browser source binding": lambda value: value["browser"].__setitem__(
                "script_sha256", "0" * 64
            ),
            "browser image binding": lambda value: value["browser"].__setitem__(
                "image_reference_sha256", "0" * 64
            ),
            "probe image binding": lambda value: value["probe"].__setitem__(
                "image_reference_sha256", "0" * 64
            ),
            "service unit binding": lambda value: value["service"].__setitem__(
                "unit_sha256", "0" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                bundle = build_bundle(Path(directory) / "bundle")
                rewrite_summary(bundle, mutate)
                with self.assertRaises(TOOL.FailureHookError):
                    TOOL.validate_bundle(bundle, TEST_BINDINGS)

    def test_rejects_browser_action_mutations(self) -> None:
        mutations = {
            "action": lambda value: value["browser_actions"][8].__setitem__(
                "action", "wait_failed"
            ),
            "boolean index": lambda value: value["browser_actions"][8].__setitem__(
                "action_index", True
            ),
            "timestamp type": lambda value: value["browser_actions"][8].__setitem__(
                "started_monotonic_ns", 340
            ),
            "noncanonical timestamp": lambda value: value["browser_actions"][
                8
            ].__setitem__("started_monotonic_ns", "0340"),
            "screenshot hash": lambda value: value["browser_actions"][4].__setitem__(
                "screenshot_sha256", "0" * 64
            ),
            "extra action field": lambda value: value["browser_actions"][8].__setitem__(
                "prompt", "must-not-escape"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                bundle = build_bundle(Path(directory) / "bundle")
                rewrite_browser(bundle, mutate)
                with self.assertRaises(TOOL.FailureHookError):
                    TOOL.validate_bundle(bundle, TEST_BINDINGS)

    def test_rejects_fault_mutations_after_rehash(self) -> None:
        mutations = {
            "boolean PID": lambda value: value.__setitem__("target_pid", True),
            "wrong worker": lambda value: value.__setitem__("target_pid", 9999),
            "wrong signal": lambda value: value.__setitem__("signal", "SIGTERM"),
            "timestamp regression": lambda value: value.__setitem__(
                "completed_monotonic_ns", 169
            ),
            "extra ID": lambda value: value.__setitem__("request_id", "secret"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                bundle = build_bundle(Path(directory) / "bundle")
                rewrite_fault(bundle, mutate)
                with self.assertRaises(TOOL.FailureHookError):
                    TOOL.validate_bundle(bundle, TEST_BINDINGS)

    def test_rejects_screenshot_and_stdout_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_bundle(Path(directory) / "bundle")
            write_file(
                bundle / "browser" / "post-header-failure.png", b"not-png", 0o400
            )
            with self.assertRaises(TOOL.FailureHookError):
                TOOL.validate_bundle(bundle, TEST_BINDINGS)
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_bundle(Path(directory) / "bundle")
            stdout_path = bundle / "browser" / "browser-stdout.jsonl"
            write_file(stdout_path, stdout_path.read_bytes() + b"{}\n", 0o600)
            with self.assertRaises(TOOL.FailureHookError):
                TOOL.validate_bundle(bundle, TEST_BINDINGS)

    def test_bundle_entry_failure_closes_open_directory_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_bundle(Path(directory) / "bundle")
            (bundle / "browser").chmod(0o755)
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(20):
                with self.assertRaises(TOOL.FailureHookError):
                    with TOOL.BundleSnapshot(bundle):
                        self.fail("invalid bundle unexpectedly opened")
            after = len(os.listdir("/proc/self/fd"))
        self.assertEqual(after, before)


class FailureHookExecutionTests(unittest.TestCase):
    def test_execute_runs_gate_then_returns_exact_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "new-bundle"
            args = args_for(bundle)

            def run(command, *, cwd, timeout):
                self.assertEqual(command[0], sys.executable)
                self.assertIn(os.fspath(bundle), command)
                self.assertEqual(cwd, ROOT)
                self.assertEqual(timeout, 390.0)
                build_bundle(bundle, bindings_for_args(args))
                return TOOL.BoundedCommandResult(0, TOOL.SUCCESS_STDOUT, b"")

            with mock.patch.object(TOOL, "run_bounded_command", side_effect=run):
                raw = TOOL.execute(args)
        lines = raw.splitlines()
        self.assertEqual(len(lines), 10)
        for line in lines:
            value = json.loads(line)
            self.assertEqual(value["schema_version"], TOOL.HOOK_SCHEMA)

    def test_execute_rejects_source_changes_after_gate_run(self) -> None:
        for changed in ("failure_gate", "browser_script"):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                gate = root / "tools" / "failure-gate.py"
                browser = root / "deploy" / "browser.cjs"
                gate.parent.mkdir(parents=True)
                browser.parent.mkdir(parents=True)
                gate.write_text("print('gate')\n", encoding="ascii")
                browser.write_text("console.log('browser');\n", encoding="ascii")
                args = args_for(root / "bundle")
                args.failure_gate = gate
                args.browser_script = browser
                original_bindings = bindings_for_args(args)

                def run(_command, *, cwd, timeout):
                    self.assertEqual(cwd, root)
                    self.assertEqual(timeout, 390.0)
                    build_bundle(args.failure_bundle, original_bindings)
                    source = getattr(args, changed)
                    source.write_bytes(source.read_bytes() + b"# changed\n")
                    return TOOL.BoundedCommandResult(0, TOOL.SUCCESS_STDOUT, b"")

                with (
                    mock.patch.object(TOOL, "run_bounded_command", side_effect=run),
                    mock.patch.object(TOOL, "validate_bundle") as validate,
                    self.assertRaises(TOOL.FailureHookError),
                ):
                    TOOL.execute(args)
                validate.assert_not_called()

    def test_execute_requires_exact_gate_process_result_before_validation(self) -> None:
        results = (
            TOOL.BoundedCommandResult(1, TOOL.SUCCESS_STDOUT, b""),
            TOOL.BoundedCommandResult(0, TOOL.SUCCESS_STDOUT + b"extra", b""),
            TOOL.BoundedCommandResult(0, TOOL.SUCCESS_STDOUT, b"warning\n"),
        )
        for result in results:
            with (
                self.subTest(result=result),
                tempfile.TemporaryDirectory() as directory,
            ):
                args = args_for(Path(directory) / "new-bundle")
                with (
                    mock.patch.object(TOOL, "run_bounded_command", return_value=result),
                    mock.patch.object(TOOL, "validate_bundle") as validate,
                    self.assertRaises(TOOL.FailureHookError),
                ):
                    TOOL.execute(args)
                validate.assert_not_called()

    def test_execute_rejects_existing_bundle_without_starting_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_bundle(Path(directory) / "bundle")
            with (
                mock.patch.object(TOOL, "run_bounded_command") as run,
                self.assertRaises(TOOL.FailureHookError),
            ):
                TOOL.execute(args_for(bundle))
            run.assert_not_called()

    def test_main_emits_no_hook_record_when_execute_fails(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        argv = [
            "--failure-gate",
            os.fspath(TOOL_PATH),
            "--failure-bundle",
            "/tmp/not-created-failure-bundle",
            "--token-file",
            "/private/token",
            "--browser-script",
            os.fspath(ROOT / "deploy/openwebui/browser-failure-smoke.cjs"),
            "--browser-image",
            "sha256:" + "a" * 64,
            "--probe-image",
            "sha256:" + "b" * 64,
            "--openwebui-url",
            "http://192.168.0.66:3000/",
            "--ready-url",
            "http://172.20.0.1:8000/readyz",
            "--network",
            "open-webui-network",
            "--service",
            "ullm-openai.service",
            "--docker",
            "docker",
            "--systemctl",
            "systemctl",
            "--journalctl",
            "journalctl",
            "--timeout-seconds",
            "360",
            "--control-timeout-ms",
            "180000",
            "--recovery-probe-timeout-seconds",
            "180",
        ]
        with (
            mock.patch.object(
                TOOL, "execute", side_effect=TOOL.FailureHookError("secret")
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(TOOL.main(argv), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "OpenWebUI failure hook failed\n")
        self.assertNotIn("secret", stderr.getvalue())

    def test_bounded_command_captures_small_output_and_rejects_overflow(self) -> None:
        result = TOOL.run_bounded_command(
            [sys.executable, "-c", "import sys;sys.stdout.write('ok')"],
            cwd=ROOT,
            timeout=2,
        )
        self.assertEqual(result, TOOL.BoundedCommandResult(0, b"ok", b""))
        with self.assertRaises(TOOL.FailureHookError):
            TOOL.run_bounded_command(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stderr.write('x'*70000);sys.stderr.flush()",
                ],
                cwd=ROOT,
                timeout=2,
            )

    def test_cli_requires_explicit_absolute_paths(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            TOOL.parse_args([])
        argv = [
            "--failure-gate",
            "relative.py",
            "--failure-bundle",
            "/tmp/bundle",
            "--token-file",
            "/tmp/token",
            "--browser-script",
            "/tmp/browser.cjs",
            "--browser-image",
            "image",
            "--probe-image",
            "probe",
            "--openwebui-url",
            "http://example.invalid/",
            "--ready-url",
            "http://example.invalid/readyz",
            "--network",
            "network",
            "--service",
            "service",
            "--docker",
            "docker",
            "--systemctl",
            "systemctl",
            "--journalctl",
            "journalctl",
            "--timeout-seconds",
            "360",
            "--control-timeout-ms",
            "180000",
            "--recovery-probe-timeout-seconds",
            "180",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            TOOL.parse_args(argv)

    def test_command_forwards_every_explicit_failure_gate_argument(self) -> None:
        args = args_for(Path("/tmp/failure-bundle"))
        command = TOOL.build_failure_gate_command(args)
        for flag in (
            "--output-dir",
            "--token-file",
            "--browser-script",
            "--browser-image",
            "--probe-image",
            "--openwebui-url",
            "--ready-url",
            "--network",
            "--service",
            "--docker",
            "--systemctl",
            "--journalctl",
            "--timeout-seconds",
            "--control-timeout-ms",
            "--recovery-probe-timeout-seconds",
        ):
            self.assertEqual(command.count(flag), 1)


if __name__ == "__main__":
    unittest.main()
