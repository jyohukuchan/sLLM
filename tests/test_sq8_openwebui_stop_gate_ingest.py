from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INGEST_SOURCE = ROOT / "tools/sq8_openwebui_stop_gate_ingest.py"
GATE_SOURCE = ROOT / "tools/run-openwebui-stop-gate.py"
BROWSER_SOURCE = ROOT / "deploy/openwebui/browser-stop-smoke.cjs"
CAMPAIGN_SOURCE = ROOT / "tools/sq8_openwebui_campaign.py"
FULL_BUNDLE_SOURCE = ROOT / "tools/sq8_full_campaign_bundle.py"
PILOT = Path("/home/homelab1/datapool/openwebui-stop-formal-pilot-20260711-180303")
PILOT_GATE_SHA256 = "b9f8c46421568abef14e49a5a2706112a0d2ca81a90ad2619232106996fb1c39"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INGEST = load_module("sq8_openwebui_stop_gate_ingest", INGEST_SOURCE)
GATE = load_module("run_openwebui_stop_gate_for_ingest_tests", GATE_SOURCE)
CAMPAIGN = load_module("sq8_openwebui_campaign", CAMPAIGN_SOURCE)
FULL_BUNDLE = load_module("sq8_full_campaign_bundle_for_stop_tests", FULL_BUNDLE_SOURCE)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_private(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def compact_line(value) -> bytes:
    return GATE.compact_json(value) + b"\n"


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return len(data).to_bytes(4, "big") + body + struct.pack(">I", zlib.crc32(body))


def minimal_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    image = zlib.compress(b"\x00\x20\x40\x60")
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", image)
        + png_chunk(b"IEND", b"")
    )


def identity(prefix: str, value: str) -> dict[str, object]:
    raw = value.encode("utf-8")
    return {f"{prefix}_utf8_bytes": len(raw), f"{prefix}_sha256": digest(raw)}


def browser_result(name: str) -> dict[str, object]:
    text = name in {"wait_visible", "click_stop", "wait_ready"}
    enabled = True if name in {"submit_chat", "click_stop", "wait_ready"} else None
    return {
        "visible": True,
        "enabled": enabled,
        "text_utf8_bytes": 12 if text else None,
        "text_sha256": digest(f"visible-{name}".encode()) if text else None,
    }


def browser_action(
    index: int,
    name: str,
    *,
    started: int,
    completed: int,
    screenshot_sha256: str,
    openwebui_url: str,
) -> dict[str, object]:
    inputs: dict[int, bytes] = {
        0: (f"{openwebui_url}/?temporary-chat=true&models={GATE.MODEL_ID}").encode(),
        1: GATE.MODEL_ID.encode(),
        2: GATE.STOP_PROMPT.encode(),
        6: GATE.RECOVERY_PROMPT.encode(),
    }
    screenshot = name == "click_stop"
    return {
        "browser_case": GATE.BROWSER_CASE,
        "action_index": index,
        "action": name,
        "selector": None,
        "input_sha256": digest(inputs[index]) if index in inputs else None,
        "started_monotonic_ns": str(started),
        "completed_monotonic_ns": str(completed),
        "result": browser_result(name),
        "screenshot_file": "browser/openwebui-stop-before.png" if screenshot else None,
        "screenshot_sha256": screenshot_sha256 if screenshot else None,
    }


def socket_event(
    sequence: int,
    target: str,
    kind: str,
    observed: int,
    *,
    done: bool = False,
    content: bool = False,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "observed_monotonic_ns": str(observed),
        "correlation_target": target,
        "type": kind,
        "done": done,
        "has_error": False,
        "content_utf8_bytes": 8 if content else 0,
        "content_sha256": digest(f"socket-{sequence}".encode()) if content else None,
    }


def lifecycle(
    name: str,
    observed: int,
    request_id: str,
    completion_id: str,
    **fields,
) -> dict[str, object]:
    return {
        "schema_version": GATE.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": observed,
        "request_id": request_id,
        "completion_id": completion_id,
        **fields,
    }


class StopBundleFixture:
    def __init__(self, parent: str | Path):
        self.parent = Path(parent)
        self.root = self.parent / "stop-bundle"
        self.browser_root = self.root / "browser"
        self.root.mkdir(parents=True)
        self.browser_root.mkdir()
        self.root.chmod(0o700)
        self.browser_root.chmod(0o700)
        self.uid = os.getuid()
        self.gid = os.getgid()
        self.secret = b"synthetic-openwebui-api-secret"
        self.openwebui_url = "http://127.0.0.1:3000"
        self.image_reference = "sha256:" + "1" * 64
        self.boot_id = "a" * 32
        self.control_group = "/system.slice/ullm-openai.service"
        self.gateway_pid = 4242
        self.worker_pid = 4243
        self.request_ids = ("request-stop-target", "request-stop-recovery")
        self.completion_ids = (
            "chatcmpl-stop-target",
            "chatcmpl-stop-recovery",
        )
        self.screenshot = minimal_png()
        self.passed = True
        self.interim, self.final = self._browser_values()
        self.observer_values = self._lifecycle_values()
        self.journal_values: list[dict[str, object]] = []
        self._write_browser()
        self._write_lifecycle()
        self.rewrite_summary()

    def _browser_values(self):
        screenshot_sha = digest(self.screenshot)
        timings = (
            (1000, 1050),
            (1060, 1110),
            (1120, 1170),
            (1180, 1380),
            (1400, 1450),
            (1460, 1580),
            (1700, 1750),
            (1760, 1860),
            (1870, 1950),
        )
        actions = [
            browser_action(
                index,
                name,
                started=timings[index][0],
                completed=timings[index][1],
                screenshot_sha256=screenshot_sha,
                openwebui_url=self.openwebui_url,
            )
            for index, name in enumerate(GATE.FINAL_ACTIONS)
        ]
        target = {
            **identity("chat_id", "chat-stop"),
            **identity("message_id", "message-stop-target"),
        }
        nonce = "e" * 64
        control_content = f"{GATE.CONTROL_SCHEMA}:{nonce}\n".encode()
        interim_events = [
            socket_event(0, "cancel_target", "chat:completion", 1350, content=True),
            socket_event(1, "cancel_target", "chat:tasks:cancel", 1505),
        ]
        interim = {
            "schema_version": GATE.BROWSER_SCHEMA,
            "record_type": "openwebui_stop_gateway_release_wait",
            "browser_case": GATE.BROWSER_CASE,
            "observed_monotonic_ns": "1600",
            "browser_actions": copy.deepcopy(actions[:6]),
            "socket_correlation": {
                "target": copy.deepcopy(target),
                "click_completed_monotonic_ns": "1450",
                "cancel_first_observed_monotonic_ns": "1505",
                "cancel_event_count": 1,
                "done_after_click_count": 0,
                "content_after_cancel_count": 0,
            },
            "socket_events": copy.deepcopy(interim_events),
            "page_error_count": 0,
            "gateway_release_control": {
                "control_schema": GATE.CONTROL_SCHEMA,
                "control_file": GATE.CONTROL_CONTAINER_PATH,
                "nonce": nonce,
                "content_utf8_bytes": len(control_content),
                "content_sha256": digest(control_content),
                "timeout_ms": 30_000,
            },
        }
        final_events = [
            *copy.deepcopy(interim_events),
            socket_event(2, "recovery_target", "chat:completion", 1810, content=True),
            socket_event(3, "recovery_target", "chat:completion", 1830, done=True),
        ]
        final = {
            "schema_version": GATE.BROWSER_SCHEMA,
            "record_type": "openwebui_stop_smoke",
            "browser_case": GATE.BROWSER_CASE,
            "observed_monotonic_ns": "2000",
            "browser_actions": actions,
            "socket_correlation": {
                "target": copy.deepcopy(target),
                "click_started_monotonic_ns": "1400",
                "click_completed_monotonic_ns": "1450",
                "cancel_first_observed_monotonic_ns": "1505",
                "cancel_event_count": 1,
                "done_after_click_count": 0,
                "content_after_cancel_count": 0,
                "recovery": {
                    **identity("chat_id", "chat-stop"),
                    **identity("message_id", "message-stop-recovery"),
                    "submit_completed_monotonic_ns": "1750",
                    "done_observed_monotonic_ns": "1830",
                    "done_event_count": 1,
                    "cancel_event_count": 0,
                },
            },
            "page_error_count": 0,
            "page_errors": [],
            "socket_events": final_events,
            "gateway_release_control": {
                "control_schema": GATE.CONTROL_SCHEMA,
                **identity("control_file", GATE.CONTROL_CONTAINER_PATH),
                "nonce_sha256": digest(nonce.encode()),
                "content_utf8_bytes": len(control_content),
                "content_sha256": digest(control_content),
                "requested_monotonic_ns": "1600",
                "observed_monotonic_ns": "1650",
            },
            "screenshot": {
                "screenshot_file": "browser/openwebui-stop-before.png",
                "screenshot_bytes": len(self.screenshot),
                "screenshot_sha256": screenshot_sha,
            },
        }
        return interim, final

    def _lifecycle_values(self):
        target_request, recovery_request = self.request_ids
        target_completion, recovery_completion = self.completion_ids
        return [
            lifecycle(
                "request_admitted",
                1100,
                target_request,
                target_completion,
                stream=True,
                prompt_tokens=32,
                max_completion_tokens=512,
            ),
            lifecycle(
                "request_started",
                1150,
                target_request,
                target_completion,
                stream=True,
                prompt_tokens=32,
                admit_to_start_ns=50,
            ),
            lifecycle(
                "request_progress",
                1200,
                target_request,
                target_completion,
                phase="prefill",
                processed_prompt_tokens=32,
                prompt_tokens=32,
            ),
            lifecycle(
                "request_first_token",
                1300,
                target_request,
                target_completion,
                stream=True,
                completion_tokens=1,
            ),
            lifecycle(
                "request_cancel_requested",
                1500,
                target_request,
                target_completion,
                stream=True,
                reason="client_disconnect",
                admit_to_cancel_ns=400,
            ),
            lifecycle(
                "request_released",
                1550,
                target_request,
                target_completion,
                stream=True,
                outcome="cancelled",
                cancel_reason="client_disconnect",
                prompt_tokens=32,
                completion_tokens=4,
                reset_complete=True,
                admit_to_start_ns=50,
                start_to_release_ns=400,
                admit_to_release_ns=450,
            ),
            lifecycle(
                "request_admitted",
                1760,
                recovery_request,
                recovery_completion,
                stream=True,
                prompt_tokens=32,
                max_completion_tokens=64,
            ),
            lifecycle(
                "request_started",
                1770,
                recovery_request,
                recovery_completion,
                stream=True,
                prompt_tokens=32,
                admit_to_start_ns=10,
            ),
            lifecycle(
                "request_progress",
                1780,
                recovery_request,
                recovery_completion,
                phase="prefill",
                processed_prompt_tokens=32,
                prompt_tokens=32,
            ),
            lifecycle(
                "request_first_token",
                1800,
                recovery_request,
                recovery_completion,
                stream=True,
                completion_tokens=1,
            ),
            lifecycle(
                "request_released",
                1840,
                recovery_request,
                recovery_completion,
                stream=True,
                outcome="stop",
                cancel_reason=None,
                prompt_tokens=32,
                completion_tokens=5,
                reset_complete=True,
                admit_to_start_ns=10,
                start_to_release_ns=70,
                admit_to_release_ns=80,
            ),
        ]

    def _write_browser(self) -> None:
        write_private(self.browser_root / "openwebui-stop-before.png", self.screenshot)
        lines = compact_line(self.interim) + compact_line(self.final)
        write_private(self.browser_root / "browser-stdout.jsonl", lines)
        write_private(
            self.browser_root / "openwebui-stop-summary.json",
            compact_line(self.final),
        )

    def _write_lifecycle(self) -> None:
        observer_raws = [GATE.compact_json(value) for value in self.observer_values]
        write_private(
            self.root / "observer.raw.jsonl",
            b"".join(raw + b"\n" for raw in observer_raws),
        )
        self.journal_values = []
        for index, (event, payload) in enumerate(
            zip(self.observer_values, observer_raws, strict=True)
        ):
            monotonic = (event["observed_monotonic_ns"] + 999) // 1000
            self.journal_values.append(
                {
                    "__CURSOR": f"stop-cursor-{index:02d}",
                    "__MONOTONIC_TIMESTAMP": str(monotonic),
                    "_BOOT_ID": self.boot_id,
                    "_PID": str(self.gateway_pid),
                    "_SYSTEMD_UNIT": GATE.SERVICE_RE.fullmatch(
                        "ullm-openai.service"
                    ).group(0),
                    "PRIORITY": "6",
                    "MESSAGE": "INFO:     " + payload.decode("ascii"),
                    "_UID": str(self.uid),
                    "_GID": str(self.gid),
                    "_SYSTEMD_CGROUP": self.control_group,
                }
            )
        write_private(
            self.root / "service-journal.raw.jsonl",
            b"".join(compact_line(value) for value in self.journal_values),
        )

    def summary_value(self) -> dict[str, object]:
        observer_raw = (self.root / "observer.raw.jsonl").read_bytes()
        journal_raw = (self.root / "service-journal.raw.jsonl").read_bytes()
        stdout_raw = (self.browser_root / "browser-stdout.jsonl").read_bytes()
        browser_summary_raw = (
            self.browser_root / "openwebui-stop-summary.json"
        ).read_bytes()
        screenshot_raw = (self.browser_root / "openwebui-stop-before.png").read_bytes()
        target_request, recovery_request = self.request_ids
        target_completion, recovery_completion = self.completion_ids
        control = self.final["gateway_release_control"]
        return {
            "schema_version": GATE.GATE_SCHEMA,
            "passed": self.passed,
            "service": {
                "unit_sha256": digest(b"ullm-openai.service"),
                "main_pid_sha256": digest(str(self.gateway_pid).encode()),
                "user_uid_sha256": digest(str(self.uid).encode()),
                "restart_count": 0,
            },
            "browser": {
                "image_sha256": digest(self.image_reference.encode()),
                "image_content_digest": self.image_reference,
                "script_sha256": digest(BROWSER_SOURCE.read_bytes()),
                "action_count": 9,
                "socket_event_count": len(self.final["socket_events"]),
                "screenshot_bytes": len(screenshot_raw),
                "screenshot_sha256": digest(screenshot_raw),
                "browser_summary_sha256": digest(browser_summary_raw),
                "stdout_lines": 2,
                "stdout_sha256": digest(stdout_raw),
                "stderr_bytes": 0,
                "stderr_sha256": digest(b""),
            },
            "gateway": {
                "request_count": 2,
                "maximum_active_requests": 1,
                "cancel_reason": "client_disconnect",
                "target_outcome": "cancelled",
                "recovery_outcome": "stop",
                "target_request_sha256": digest(target_request.encode()),
                "target_completion_sha256": digest(target_completion.encode()),
                "recovery_request_sha256": digest(recovery_request.encode()),
                "recovery_completion_sha256": digest(recovery_completion.encode()),
                "control_content_sha256": control["content_sha256"],
            },
            "artifacts": {
                "observer": {
                    "file": "observer.raw.jsonl",
                    "bytes": len(observer_raw),
                    "records": len(self.observer_values),
                    "sha256": digest(observer_raw),
                },
                "journal": {
                    "file": "service-journal.raw.jsonl",
                    "bytes": len(journal_raw),
                    "records": len(self.journal_values),
                    "sha256": digest(journal_raw),
                    "unique_cursors": len(
                        {value["__CURSOR"] for value in self.journal_values}
                    ),
                    "stderr_bytes": 0,
                    "stderr_sha256": digest(b""),
                },
            },
        }

    def rewrite_summary(self) -> None:
        self.summary = self.summary_value()
        write_private(self.root / "summary.json", compact_line(self.summary))

    def rewrite_browser(self) -> None:
        self._write_browser()
        self.rewrite_summary()

    def rewrite_lifecycle(self) -> None:
        self._write_lifecycle()
        self.rewrite_summary()

    def bindings(self):
        return INGEST.StopGateInputBindings(
            gate_source=GATE_SOURCE,
            gate_source_sha256=digest(GATE_SOURCE.read_bytes()),
            browser_script=BROWSER_SOURCE,
            browser_script_sha256=digest(BROWSER_SOURCE.read_bytes()),
            browser_image_reference=self.image_reference,
            browser_image_content_id=self.image_reference,
            openwebui_url=self.openwebui_url,
            service_unit="ullm-openai.service",
            service_user="homelab1",
            boot_id=self.boot_id,
            control_group=self.control_group,
            gateway_pid=self.gateway_pid,
            gateway_starttime_ticks=10_001,
            worker_pid=self.worker_pid,
            worker_starttime_ticks=10_002,
            restart_count=0,
            uid=self.uid,
            gid=self.gid,
            forbidden_values=(self.secret,),
        )


def replace_dataclass(value, **changes):
    fields = {
        field.name: getattr(value, field.name)
        for field in value.__dataclass_fields__.values()
    }
    fields.update(changes)
    return type(value)(**fields)


class StopGatePublicContractTests(unittest.TestCase):
    def test_public_contract_is_exposed(self):
        self.assertTrue(hasattr(INGEST, "StopGateInputBindings"))
        self.assertTrue(hasattr(INGEST, "StopGateIngestResult"))
        self.assertTrue(hasattr(INGEST, "ingest_stop_gate_bundle"))


class StopGateIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = StopBundleFixture(self.temporary.name)

    def fresh(self, name: str) -> StopBundleFixture:
        self.fixture = StopBundleFixture(Path(self.temporary.name) / name)
        return self.fixture

    def ingest(self, bindings=None):
        return INGEST.ingest_stop_gate_bundle(
            self.fixture.root,
            self.fixture.bindings() if bindings is None else bindings,
        )

    def test_positive_converts_actions_claims_screenshot_and_redacted_view(self):
        actions, claims, screenshot, view = self.ingest()
        self.assertEqual(len(actions), 9)
        self.assertEqual(
            [record["fields"]["action_index"] for record in actions], list(range(9))
        )
        self.assertTrue(all(record["phase"] == "cancellation" for record in actions))
        self.assertEqual(
            [record["case_id"] for record in actions],
            ["openwebui_stop_after_visible_content"] * 6
            + ["openwebui_stop_after_visible_content-recovery"] * 3,
        )
        self.assertTrue(
            all(
                type(record["fields"]["started_monotonic_ns"]) is int
                and type(record["fields"]["completed_monotonic_ns"]) is int
                for record in actions
            )
        )
        self.assertEqual(len(claims), 11)
        self.assertEqual(
            [claim.case_id for claim in claims],
            ["openwebui_stop_after_visible_content"] * 6
            + ["openwebui_stop_after_visible_content-recovery"] * 5,
        )
        self.assertTrue(all(claim.phase == "cancellation" for claim in claims))
        for claim in claims:
            self.assertEqual(
                set(json.loads(claim.raw)), set(INGEST.REQUIRED_JOURNAL_FIELDS)
            )
        self.assertEqual(screenshot.path.resolve(), screenshot.path)
        self.assertEqual(screenshot.bytes, len(self.fixture.screenshot))
        self.assertEqual(screenshot.sha256, digest(self.fixture.screenshot))
        encoded = GATE.compact_json(view)
        self.assertNotIn(b"passed", encoded)
        for forbidden in (
            self.fixture.secret,
            self.fixture.openwebui_url.encode(),
            self.fixture.request_ids[0].encode(),
            self.fixture.completion_ids[0].encode(),
            self.fixture.interim["gateway_release_control"]["nonce"].encode(),
            os.fspath(GATE_SOURCE).encode(),
            GATE.STOP_PROMPT.encode(),
        ):
            self.assertNotIn(forbidden, encoded)

    def test_false_producer_verdict_is_not_a_trusted_input(self):
        self.fixture.passed = False
        self.fixture.rewrite_summary()
        result = self.ingest()
        self.assertEqual(len(result.lifecycle_claims), 11)
        self.assertNotIn("passed", result.derived_view)

    def test_screenshot_contract_stream_copies_identical_bytes(self):
        result = self.ingest()
        destination_parent = Path(self.temporary.name) / "campaign"
        destination_parent.mkdir()
        campaign = FULL_BUNDLE.AtomicCampaignDirectory(
            destination_parent / "final",
            uid=os.getuid(),
            gid=os.getgid(),
        )
        self.addCleanup(campaign.abort)
        evidence = campaign.copy_file(
            result.screenshot_evidence.path,
            "browser/openwebui-stop-before.png",
            expected_bytes=result.screenshot_evidence.bytes,
            expected_sha256=result.screenshot_evidence.sha256,
            maximum_bytes=64 << 20,
            scan=lambda _raw, _label: None,
        )
        self.assertEqual(evidence.bytes, result.screenshot_evidence.bytes)
        self.assertEqual(evidence.sha256, result.screenshot_evidence.sha256)
        self.assertEqual(
            (campaign.stage_path / "browser/openwebui-stop-before.png").read_bytes(),
            self.fixture.screenshot,
        )

    def test_claims_are_consumed_by_global_campaign_capture(self):
        claims = self.ingest().lifecycle_claims

        class StaticJournalSource:
            def __init__(self, rows):
                self.rows = iter(rows)

            def open_after(self, _unit, _boot_id):
                return "stop-campaign-anchor"

            def read_next(self, timeout_usec):
                try:
                    return next(self.rows)
                except StopIteration:
                    time.sleep(min(timeout_usec / 1_000_000, 0.001))
                    return None

            def close(self):
                return None

        capture = CAMPAIGN.CampaignJournalCapture(
            Path(self.temporary.name) / "campaign-journal.raw.jsonl",
            self.fixture.boot_id,
            CAMPAIGN.PidEpoch(self.fixture.gateway_pid, self.fixture.worker_pid),
            scan_raw=lambda _raw, _label: None,
            source=StaticJournalSource([claim.raw for claim in claims]),
        )
        self.addCleanup(capture.abort)
        capture.start()
        consumed = capture.claim_bundle_records(
            claims, time.monotonic_ns() + 2_000_000_000
        )
        self.assertEqual(len(consumed), 11)
        self.assertEqual(
            [item.case_id for item in consumed],
            ["openwebui_stop_after_visible_content"] * 6
            + ["openwebui_stop_after_visible_content-recovery"] * 5,
        )

    def test_browser_interim_final_action_and_summary_mutations_are_rejected(self):
        self.fixture.interim["page_error_count"] = 1
        self.fixture.rewrite_browser()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("action")
        self.fixture.interim["browser_actions"][4]["result"]["enabled"] = False
        self.fixture.final["browser_actions"][4]["result"]["enabled"] = False
        self.fixture.rewrite_browser()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("summary")
        self.fixture.summary["gateway"]["request_count"] = 1
        write_private(
            self.fixture.root / "summary.json", compact_line(self.fixture.summary)
        )
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("browser-summary")
        write_private(
            self.fixture.browser_root / "openwebui-stop-summary.json", b"{}\n"
        )
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

    def test_screenshot_wrong_hash_and_crc_mutations_are_rejected(self):
        self.fixture.final["screenshot"]["screenshot_sha256"] = "0" * 64
        self.fixture.final["browser_actions"][4]["screenshot_sha256"] = "0" * 64
        self.fixture.rewrite_browser()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("png-crc")
        corrupted = bytearray(self.fixture.screenshot)
        idat = corrupted.index(b"IDAT")
        corrupted[idat + 4] ^= 1
        self.fixture.screenshot = bytes(corrupted)
        screenshot_sha = digest(self.fixture.screenshot)
        self.fixture.final["screenshot"]["screenshot_bytes"] = len(
            self.fixture.screenshot
        )
        self.fixture.final["screenshot"]["screenshot_sha256"] = screenshot_sha
        self.fixture.final["browser_actions"][4]["screenshot_sha256"] = screenshot_sha
        self.fixture.interim["browser_actions"][4]["screenshot_sha256"] = screenshot_sha
        self.fixture._write_browser()
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

    def test_lifecycle_semantics_observer_and_journal_mutations_are_rejected(self):
        cancel = self.fixture.observer_values[4]
        cancel["reason"] = "server_shutdown"
        self.fixture.rewrite_lifecycle()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("duration")
        self.fixture.observer_values[5]["admit_to_release_ns"] += 1
        self.fixture.rewrite_lifecycle()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("deadline")
        self.fixture.observer_values[5]["observed_monotonic_ns"] = (
            self.fixture.observer_values[4]["observed_monotonic_ns"] + 5_000_000_001
        )
        for index in range(6, 11):
            self.fixture.observer_values[index]["observed_monotonic_ns"] += (
                5_000_000_001
            )
        self.fixture.rewrite_lifecycle()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("observer-journal")
        self.fixture.journal_values[0]["MESSAGE"] = self.fixture.journal_values[1][
            "MESSAGE"
        ]
        write_private(
            self.fixture.root / "service-journal.raw.jsonl",
            b"".join(compact_line(value) for value in self.fixture.journal_values),
        )
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("priority")
        del self.fixture.journal_values[0]["PRIORITY"]
        write_private(
            self.fixture.root / "service-journal.raw.jsonl",
            b"".join(compact_line(value) for value in self.fixture.journal_values),
        )
        self.fixture.rewrite_summary()
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

    def test_layout_mode_symlink_hardlink_and_extra_are_rejected(self):
        write_private(self.fixture.root / "unexpected", b"unexpected\n")
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("browser-extra")
        write_private(self.fixture.browser_root / "unexpected", b"unexpected\n")
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("root-mode")
        self.fixture.root.chmod(0o750)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("mode")
        (self.fixture.root / "summary.json").chmod(0o640)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("browser-mode")
        self.fixture.browser_root.chmod(0o750)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("owner-binding")
        bindings = replace_dataclass(self.fixture.bindings(), uid=self.fixture.uid + 1)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest(bindings)

        self.fresh("hardlink")
        os.link(
            self.fixture.root / "summary.json",
            self.fixture.root.parent / "summary-hardlink",
        )
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        self.fresh("symlink")
        summary = self.fixture.root / "summary.json"
        outside = self.fixture.root.parent / "outside-summary"
        summary.rename(outside)
        summary.symlink_to(outside)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

    def test_source_binding_secret_boundary_and_source_toctou_are_rejected(self):
        bindings = replace_dataclass(
            self.fixture.bindings(), gate_source_sha256="0" * 64
        )
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest(bindings)

        bindings = replace_dataclass(
            self.fixture.bindings(), browser_script_sha256="0" * 64
        )
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest(bindings)

        self.fresh("secret")
        path = self.fixture.root / "summary.json"
        raw = path.read_bytes()
        padding = b" " * (INGEST.COPY_CHUNK_BYTES - 3 - len(raw))
        write_private(path, raw + padding + self.fixture.secret)
        with self.assertRaises(INGEST.StopGateIngestError):
            self.ingest()

        source_root = Path(self.temporary.name) / "source-toctou"
        source_root.mkdir()
        source = source_root / "source.py"
        write_private(source, b"source bytes\n")
        stable = INGEST._StableSource(
            source,
            "test source",
            1024,
            digest(source.read_bytes()),
            (self.fixture.secret,),
        )
        self.addCleanup(stable.close)
        replacement = source_root / "replacement"
        write_private(replacement, source.read_bytes())
        os.replace(replacement, source)
        with self.assertRaises(INGEST.StopGateIngestError):
            stable.seal()

    def test_bundle_toctou_replacement_is_rejected_at_seal(self):
        snapshot = INGEST._BundleSnapshot(
            self.fixture.root,
            uid=self.fixture.uid,
            gid=self.fixture.gid,
            forbidden_values=(self.fixture.secret,),
        )
        self.addCleanup(snapshot.close)
        list(snapshot.iter_lines("browser_stdout"))
        snapshot.read_small("browser_summary")
        snapshot.consume_png()
        list(snapshot.iter_lines("observer"))
        list(snapshot.iter_lines("journal"))
        snapshot.read_small("summary")
        target = self.fixture.browser_root / "openwebui-stop-before.png"
        replacement = self.fixture.browser_root / "replacement"
        write_private(replacement, target.read_bytes())
        os.replace(replacement, target)
        with self.assertRaises(INGEST.StopGateIngestError):
            snapshot.seal()

    def test_actual_pilot_revalidates_only_with_matching_sources(self):
        if not PILOT.is_dir():
            self.skipTest("formal OpenWebUI Stop pilot is unavailable")
        gate_sha = digest(GATE_SOURCE.read_bytes())
        summary = json.loads((PILOT / "summary.json").read_bytes())
        browser_sha = digest(BROWSER_SOURCE.read_bytes())
        if (
            gate_sha != PILOT_GATE_SHA256
            or browser_sha != summary["browser"]["script_sha256"]
        ):
            self.skipTest(
                "formal Stop pilot source binding differs from current source"
            )
        first_journal = json.loads(
            (PILOT / "service-journal.raw.jsonl").read_text().splitlines()[0]
        )
        final_browser = json.loads(
            (PILOT / "browser/openwebui-stop-summary.json").read_bytes()
        )
        openwebui_url = "http://127.0.0.1:3000"
        navigation = (
            f"{openwebui_url}/?temporary-chat=true&models={GATE.MODEL_ID}"
        ).encode()
        if final_browser["browser_actions"][0]["input_sha256"] != digest(navigation):
            self.skipTest("formal Stop pilot OpenWebUI URL binding is unavailable")
        image = summary["browser"]["image_content_digest"]
        gateway_pid = int(first_journal["_PID"])
        bindings = INGEST.StopGateInputBindings(
            gate_source=GATE_SOURCE,
            gate_source_sha256=gate_sha,
            browser_script=BROWSER_SOURCE,
            browser_script_sha256=browser_sha,
            browser_image_reference=image,
            browser_image_content_id=image,
            openwebui_url=openwebui_url,
            service_unit=first_journal["_SYSTEMD_UNIT"],
            service_user="homelab1",
            boot_id=first_journal["_BOOT_ID"],
            control_group=first_journal["_SYSTEMD_CGROUP"],
            gateway_pid=gateway_pid,
            gateway_starttime_ticks=1,
            worker_pid=gateway_pid + 1,
            worker_starttime_ticks=1,
            restart_count=summary["service"]["restart_count"],
            uid=int(first_journal["_UID"]),
            gid=int(first_journal["_GID"]),
            forbidden_values=(b"formal-stop-pilot-secret-sentinel",),
        )
        result = INGEST.ingest_stop_gate_bundle(PILOT, bindings)
        self.assertEqual(len(result.browser_action_records), 9)
        self.assertEqual(len(result.lifecycle_claims), 11)
        self.assertEqual(result.screenshot_evidence.bytes, 41_093)


if __name__ == "__main__":
    unittest.main()
