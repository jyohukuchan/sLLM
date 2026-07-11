from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_openwebui_failure_gate_ingest.py"
HOOK_TEST_PATH = ROOT / "tests" / "test_run_openwebui_failure_hook.py"
CAMPAIGN_PATH = ROOT / "tools" / "sq8_openwebui_campaign.py"
GATE_SOURCE = ROOT / "tools" / "run-openwebui-failure-gate.py"
HOOK_SOURCE = ROOT / "tools" / "run-openwebui-failure-hook.py"
BROWSER_SOURCE = ROOT / "deploy" / "openwebui" / "browser-failure-smoke.cjs"
PILOT = Path("/home/homelab1/datapool/openwebui-failure-formal-20260711-195916")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INGEST = load_module("sq8_openwebui_failure_gate_ingest", MODULE_PATH)
HOOK_FIXTURE = load_module("failure_hook_fixture_for_ingest", HOOK_TEST_PATH)
campaign_existing = sys.modules.get("sq8_openwebui_campaign")
if (
    campaign_existing is None
    or Path(campaign_existing.__file__).resolve() != CAMPAIGN_PATH.resolve()
):
    CAMPAIGN = load_module("sq8_openwebui_campaign", CAMPAIGN_PATH)
else:
    CAMPAIGN = campaign_existing


def digest(value: str | bytes | Path) -> str:
    if isinstance(value, Path):
        raw = value.read_bytes()
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = value
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


def write_file(path: Path, raw: bytes, mode: int) -> None:
    if path.exists() and not path.is_symlink():
        path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + (zlib.crc32(kind + payload) & 0xFFFF_FFFF).to_bytes(4, "big")
    )


VALID_PNG = b"".join(
    (
        b"\x89PNG\r\n\x1a\n",
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(b"\x00\x11\x22\x33")),
        png_chunk(b"IEND", b""),
    )
)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def lifecycle(
    event: str, timestamp: int, request: str, completion: str, **fields: object
) -> dict:
    return {
        "schema_version": "ullm.gateway.lifecycle.v1",
        "event": event,
        "observed_monotonic_ns": timestamp,
        "request_id": request,
        "completion_id": completion,
        **fields,
    }


def lifecycle_events() -> list[dict]:
    old_request, old_completion = "target-request", "target-completion"
    new_request, new_completion = "recovery-request", "recovery-completion"
    return [
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
            175,
            old_request,
            old_completion,
            reason="unexpected worker stdout EOF",
            admit_to_fatal_ns=75,
        ),
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


class SyntheticBundle:
    BOOT_ID = "1" * 32
    NETWORK_ID = "2" * 64
    BROWSER_IMAGE = "sha256:" + "3" * 64
    PROBE_IMAGE = "sha256:" + "4" * 64
    SECRET = b"synthetic-secret-sentinel"

    def __init__(self, root: Path):
        self.root = root

    def bindings(self, *, forbidden_values: tuple[bytes, ...] | None = None):
        return INGEST.FailureGateInputBindings(
            gate_source=GATE_SOURCE,
            gate_source_sha256=digest(GATE_SOURCE),
            hook_source=HOOK_SOURCE,
            hook_source_sha256=digest(HOOK_SOURCE),
            browser_source=BROWSER_SOURCE,
            browser_source_sha256=digest(BROWSER_SOURCE),
            browser_image_reference=self.BROWSER_IMAGE,
            browser_image_content_digest="sha256:" + digest("browser-image"),
            probe_image_reference=self.PROBE_IMAGE,
            probe_image_content_digest="sha256:" + digest("probe-image"),
            docker_network_id=self.NETWORK_ID,
            docker_network_subnet="172.20.0.0/16",
            docker_network_gateway="172.20.0.1",
            service_unit="ullm-openai.service",
            service_user="homelab1",
            boot_id=self.BOOT_ID,
            control_group="/system.slice/ullm-openai.service",
            normal_gateway_pid=1001,
            normal_gateway_starttime_ticks=9001,
            normal_worker_pid=1002,
            normal_worker_starttime_ticks=9002,
            normal_restart_count=3,
            restart_gateway_pid=2001,
            restart_gateway_starttime_ticks=9901,
            restart_worker_pid=2002,
            restart_worker_starttime_ticks=9902,
            restart_restart_count=4,
            uid=os.getuid(),
            gid=os.getgid(),
            forbidden_values=(self.SECRET,)
            if forbidden_values is None
            else forbidden_values,
        )

    def build(self, name: str = "bundle") -> Path:
        path = self.root / name
        bindings = self.bindings()
        hook_bindings = HOOK_FIXTURE.TOOL.BundleBindings(
            gate_source_sha256=bindings.gate_source_sha256,
            browser_script_sha256=bindings.browser_source_sha256,
            browser_image_reference_sha256=digest(bindings.browser_image_reference),
            probe_image_reference_sha256=digest(bindings.probe_image_reference),
            service_unit_sha256=digest(bindings.service_unit),
        )
        HOOK_FIXTURE.build_bundle(path, hook_bindings)

        self.replace_screenshot(path, VALID_PNG)

        readiness = load_json(path / "readiness-evidence.json")
        readiness["network_id"] = bindings.docker_network_id
        readiness["subnet"] = bindings.docker_network_subnet
        readiness["gateway"] = bindings.docker_network_gateway
        readiness_raw = canonical(readiness)
        write_file(path / "readiness-evidence.json", readiness_raw, 0o600)

        rows = []
        for index, event in enumerate(lifecycle_events()):
            payload = canonical(event)[:-1].decode("ascii")
            pid = (
                bindings.normal_gateway_pid
                if index < 5
                else bindings.restart_gateway_pid
            )
            rows.append(
                {
                    "__CURSOR": f"s=synthetic-{index}",
                    "__MONOTONIC_TIMESTAMP": str(1000 + index),
                    "_BOOT_ID": bindings.boot_id,
                    "_PID": str(pid),
                    "_SYSTEMD_UNIT": bindings.service_unit,
                    "_UID": str(bindings.uid),
                    "_GID": str(bindings.gid),
                    "PRIORITY": "6",
                    "MESSAGE": "INFO:     " + payload,
                }
            )
        journal_raw = b"".join(canonical(row) for row in rows)
        write_file(path / "service-journal.raw.jsonl", journal_raw, 0o600)

        summary = load_json(path / "summary.json")
        summary["passed"] = False
        summary["service"]["boot_id_sha256"] = digest(bindings.boot_id)
        summary["fault"].update(
            {
                "target_request_sha256": digest("target-request"),
                "target_completion_sha256": digest("target-completion"),
                "worker_fatal_monotonic_ns": 175,
                "signal_to_fatal_ns": 5,
            }
        )
        summary["recovery"].update(
            {
                "request_sha256": digest("recovery-request"),
                "completion_sha256": digest("recovery-completion"),
                "admitted_monotonic_ns": 310,
                "released_monotonic_ns": 334,
                "readiness_artifact_sha256": digest(readiness_raw),
            }
        )
        summary["gateway_journal"].update(
            {
                "lifecycle_count": 10,
                "record_count": 10,
                "cursor_count": 10,
                "raw_sha256": digest(journal_raw),
            }
        )
        summary["probe"]["network_id_sha256"] = digest(bindings.docker_network_id)
        write_file(path / "summary.json", canonical(summary), 0o600)
        return path

    @staticmethod
    def refresh_summary(path: Path, mutate) -> None:
        summary = load_json(path / "summary.json")
        mutate(summary)
        write_file(path / "summary.json", canonical(summary), 0o600)

    def rewrite_browser(self, path: Path, mutate) -> None:
        browser_path = path / "browser/openwebui-failure-summary.json"
        browser = load_json(browser_path)
        mutate(browser)
        final_raw = (
            json.dumps(browser, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
            )
            + b"\n"
        )
        stdout_path = path / "browser/browser-stdout.jsonl"
        lines = stdout_path.read_bytes().splitlines(keepends=True)
        stdout_raw = b"".join(lines[:2]) + final_raw
        write_file(browser_path, final_raw, 0o400)
        write_file(stdout_path, stdout_raw, 0o600)

        def update(summary: dict) -> None:
            summary["browser"]["stdout_bytes"] = len(stdout_raw)
            summary["browser"]["stdout_sha256"] = digest(stdout_raw)
            summary["browser"]["screenshot_sha256"] = browser["screenshot"][
                "screenshot_sha256"
            ]

        self.refresh_summary(path, update)

    def rewrite_fault(self, path: Path, mutate) -> None:
        fault_path = path / "fault-injection.json"
        fault = load_json(fault_path)
        mutate(fault)
        raw = canonical(fault)
        write_file(fault_path, raw, 0o600)
        self.refresh_summary(
            path,
            lambda summary: summary["fault"].__setitem__(
                "fault_artifact_sha256", digest(raw)
            ),
        )

    def rewrite_readiness(self, path: Path, mutate) -> None:
        readiness_path = path / "readiness-evidence.json"
        readiness = load_json(readiness_path)
        mutate(readiness)
        raw = canonical(readiness)
        write_file(readiness_path, raw, 0o600)

        def update(summary: dict) -> None:
            summary["recovery"]["readiness_artifact_sha256"] = digest(raw)
            summary["probe"]["network_id_sha256"] = digest(readiness["network_id"])

        self.refresh_summary(path, update)

    def rewrite_journal(self, path: Path, mutate) -> None:
        journal_path = path / "service-journal.raw.jsonl"
        rows = [json.loads(raw) for raw in journal_path.read_text().splitlines()]
        mutate(rows)
        raw = b"".join(canonical(row) for row in rows)
        write_file(journal_path, raw, 0o600)

        def update(summary: dict) -> None:
            summary["gateway_journal"]["record_count"] = len(rows)
            summary["gateway_journal"]["cursor_count"] = len(rows)
            summary["gateway_journal"]["raw_sha256"] = digest(raw)

        self.refresh_summary(path, update)

    def replace_screenshot(self, path: Path, raw: bytes) -> None:
        screenshot_sha = digest(raw)
        write_file(path / "browser/post-header-failure.png", raw, 0o400)
        stdout_path = path / "browser/browser-stdout.jsonl"
        records = [json.loads(line) for line in stdout_path.read_text().splitlines()]
        records[1]["browser_actions"][4]["screenshot_sha256"] = screenshot_sha
        final = records[2]
        final["screenshot"]["screenshot_bytes"] = len(raw)
        final["screenshot"]["screenshot_sha256"] = screenshot_sha
        final["browser_actions"][4]["screenshot_sha256"] = screenshot_sha
        stdout_raw = b"".join(
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
            for value in records
        )
        final_raw = (
            json.dumps(final, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
        )
        write_file(stdout_path, stdout_raw, 0o600)
        write_file(path / "browser/openwebui-failure-summary.json", final_raw, 0o400)

        def update(summary: dict) -> None:
            summary["browser"]["stdout_bytes"] = len(stdout_raw)
            summary["browser"]["stdout_sha256"] = digest(stdout_raw)
            summary["browser"]["screenshot_sha256"] = screenshot_sha

        self.refresh_summary(path, update)


class FakeJournalSource:
    def __init__(self, boot_id: str):
        self.boot_id = boot_id
        self.condition = threading.Condition()
        self.queue: deque[bytes] = deque()
        self.closed = False
        self.owner_thread: int | None = None

    def open_after(self, unit: str, boot_id: str) -> str:
        if unit != CAMPAIGN.SERVICE_UNIT or boot_id != self.boot_id:
            raise AssertionError("fake journal identity differs")
        self.owner_thread = threading.get_ident()
        return "anchor-cursor"

    def read_next(self, timeout_usec: int) -> bytes | None:
        if threading.get_ident() != self.owner_thread:
            raise AssertionError("fake journal escaped its reader thread")
        deadline = time.monotonic() + timeout_usec / 1_000_000
        with self.condition:
            while not self.queue and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.closed:
                return None
            return self.queue.popleft()

    def close(self) -> None:
        if self.owner_thread is not None and threading.get_ident() != self.owner_thread:
            raise AssertionError("fake journal close escaped its reader thread")
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def feed(self, *rows: bytes) -> None:
        with self.condition:
            self.queue.extend(rows)
            self.condition.notify_all()


class FailureGateIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = SyntheticBundle(self.root)

    def ingest(self, path: Path, bindings=None):
        return INGEST.ingest_failure_gate_bundle(
            path, self.fixture.bindings() if bindings is None else bindings
        )

    def test_synthetic_bundle_is_reconstructed_without_trusting_passed(self) -> None:
        bundle = self.fixture.build()
        result = self.ingest(bundle)
        self.assertEqual(len(result.browser_action_records), 9)
        self.assertEqual(
            result.fault_injection_record["case_id"], "post-header-failure"
        )
        self.assertEqual(len(result.lifecycle_claims), 10)
        self.assertEqual(
            [claim.case_id for claim in result.lifecycle_claims],
            ["post-header-failure"] * 5 + ["post-header-recovery"] * 5,
        )
        self.assertTrue(
            all(
                claim.phase == "post_header_failure"
                for claim in result.lifecycle_claims
            )
        )
        self.assertEqual(
            result.restart_probe_record["fields"]["probe"],
            "post-header-restart-ready",
        )
        self.assertEqual(
            result.screenshot_evidence.source_path,
            bundle / result.screenshot_evidence.bundle_path,
        )
        self.assertEqual(result.derived_view["journal"]["lifecycle_count"], 10)
        self.assertFalse(load_json(bundle / "summary.json")["passed"])

    def test_global_campaign_capture_consumes_all_claims_across_restart(self) -> None:
        result = self.ingest(self.fixture.build())
        source = FakeJournalSource(self.fixture.BOOT_ID)
        capture = CAMPAIGN.CampaignJournalCapture(
            self.root / "campaign-journal.jsonl",
            self.fixture.BOOT_ID,
            CAMPAIGN.PidEpoch(1001, 1002),
            scan_raw=lambda _raw, _label: None,
            source=source,
        )
        self.addCleanup(capture.abort)
        self.assertEqual(capture.start(), "anchor-cursor")
        deadline = time.monotonic_ns() + 3_000_000_000
        capture.checkpoint("resource_normal", deadline)
        capture.arm_restart_transition()
        source.feed(*(claim.raw for claim in result.lifecycle_claims))
        claimed = capture.claim_bundle_records(result.lifecycle_claims, deadline)
        capture.confirm_restart_epoch(CAMPAIGN.PidEpoch(2001, 2002))
        capture.checkpoint("post_header_failure", deadline)
        self.assertEqual(len(claimed), 10)
        self.assertEqual(
            [item.case_id for item in claimed],
            ["post-header-failure"] * 5 + ["post-header-recovery"] * 5,
        )
        self.assertEqual(claimed[4].fields["event"]["event"], "worker_fatal")
        self.assertEqual(claimed[-1].fields["event"]["event"], "request_released")

    def test_raw_action_fault_readiness_and_journal_mutations_are_rejected(
        self,
    ) -> None:
        cases = {
            "raw summary": lambda path: self.fixture.refresh_summary(
                path, lambda value: value.__setitem__("untrusted", True)
            ),
            "browser action": lambda path: self.fixture.rewrite_browser(
                path,
                lambda value: value["browser_actions"][8].__setitem__(
                    "action", "wait_failed"
                ),
            ),
            "fault": lambda path: self.fixture.rewrite_fault(
                path, lambda value: value.__setitem__("target_parent_pid", 9999)
            ),
            "readiness": lambda path: self.fixture.rewrite_readiness(
                path,
                lambda value: value["recovered"].__setitem__("status", 503),
            ),
            "journal": self._mutate_fatal_reason,
        }
        for index, (name, mutate) in enumerate(cases.items()):
            with self.subTest(name=name):
                bundle = self.fixture.build(f"semantic-{index}")
                mutate(bundle)
                with self.assertRaises(INGEST.FailureGateIngestError):
                    self.ingest(bundle)

    def _mutate_fatal_reason(self, path: Path) -> None:
        def mutate(rows: list[dict]) -> None:
            row = rows[4]
            prefix = "INFO:     "
            event = json.loads(row["MESSAGE"][len(prefix) :])
            event["reason"] = "different reason"
            row["MESSAGE"] = prefix + canonical(event)[:-1].decode("ascii")

        self.fixture.rewrite_journal(path, mutate)

    def test_rehashed_png_framing_truncation_and_crc_mutations_are_rejected(
        self,
    ) -> None:
        corrupt_crc = bytearray(VALID_PNG)
        corrupt_crc[29] ^= 0x01
        cases = {
            "not PNG": b"not-a-png-but-rehashed",
            "truncated": VALID_PNG[:-3],
            "CRC": bytes(corrupt_crc),
        }
        for index, (name, screenshot) in enumerate(cases.items()):
            with self.subTest(name=name):
                bundle = self.fixture.build(f"png-negative-{index}")
                self.fixture.replace_screenshot(bundle, screenshot)
                with self.assertRaises(INGEST.FailureGateIngestError):
                    self.ingest(bundle)

    def test_layout_link_mode_owner_and_symlink_mutations_are_rejected(self) -> None:
        mutations = {
            "extra": lambda path: write_file(path / "extra", b"x", 0o600),
            "mode": lambda path: (path / "summary.json").chmod(0o644),
            "hardlink": lambda path: os.link(
                path / "browser/post-header-failure.png", self.root / "hardlink"
            ),
            "symlink": self._replace_summary_with_symlink,
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name=name):
                bundle = self.fixture.build(f"layout-{index}")
                mutate(bundle)
                with self.assertRaises(INGEST.FailureGateIngestError):
                    self.ingest(bundle)
        owner_bundle = self.fixture.build("owner")
        wrong_owner = dataclasses.replace(self.fixture.bindings(), uid=os.getuid() + 1)
        with self.assertRaises(INGEST.FailureGateIngestError):
            self.ingest(owner_bundle, wrong_owner)

    def _replace_summary_with_symlink(self, path: Path) -> None:
        external = self.root / "external-summary"
        external.write_bytes((path / "summary.json").read_bytes())
        (path / "summary.json").unlink()
        (path / "summary.json").symlink_to(external)

    def test_source_hash_and_source_toctou_are_rejected(self) -> None:
        bundle = self.fixture.build()
        wrong = dataclasses.replace(
            self.fixture.bindings(), gate_source_sha256="0" * 64
        )
        with self.assertRaises(INGEST.FailureGateIngestError):
            self.ingest(bundle, wrong)

        source_dir = self.root / "source"
        source_dir.mkdir()
        source_path = source_dir / GATE_SOURCE.name
        shutil.copyfile(GATE_SOURCE, source_path)
        source_path.chmod(0o600)
        stable = INGEST._StableSource(
            source_path,
            label="test failure source",
            maximum=4 << 20,
            expected_sha256=digest(source_path),
            forbidden_values=(),
        )
        try:
            replacement = source_dir / "replacement"
            replacement.write_bytes(source_path.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, source_path)
            with self.assertRaises(INGEST.FailureGateIngestError):
                stable.seal()
        finally:
            stable.close()

    def test_bundle_replacement_after_streaming_is_rejected_at_seal(self) -> None:
        bundle = self.fixture.build()
        snapshot = INGEST._BundleSnapshot(
            bundle,
            uid=os.getuid(),
            gid=os.getgid(),
            forbidden_values=(),
        )
        try:
            for key in tuple(snapshot.files):
                snapshot.stream(key)
            replacement = self.root / "replacement-summary"
            replacement.write_bytes((bundle / "summary.json").read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, bundle / "summary.json")
            with self.assertRaises(INGEST.FailureGateIngestError):
                snapshot.seal()
        finally:
            snapshot.close()

    def test_forbidden_secret_split_across_stream_chunks_is_rejected(self) -> None:
        bundle = self.fixture.build()
        secret = b"forbidden-split-secret"
        raw = b"x" * (INGEST.COPY_CHUNK_BYTES - 3) + secret + b"\n"
        write_file(bundle / "summary.json", raw, 0o600)
        bindings = self.fixture.bindings(forbidden_values=(secret,))
        with self.assertRaises(INGEST.FailureGateIngestError):
            self.ingest(bundle, bindings)

    @unittest.skipUnless(PILOT.is_dir(), "formal failure pilot is unavailable")
    def test_formal_pilot_matches_pinned_current_sources(self) -> None:
        self.assertEqual(
            digest(GATE_SOURCE),
            "63075654ef80c6a165a133a3d9a73a21477f0c6e0d6412d0adc02f516807c544",
        )
        self.assertEqual(
            digest(HOOK_SOURCE),
            "fbc1ea6d3cfd314cda55b6665fb3bf67f5c90731a4d21158fe9df3814e145dfd",
        )
        self.assertEqual(
            digest(BROWSER_SOURCE),
            "ff08666d8917d5cf134e886ed61916312b6e8894c010b762aca531d47fa391bb",
        )
        bindings = INGEST.FailureGateInputBindings(
            gate_source=GATE_SOURCE,
            gate_source_sha256=digest(GATE_SOURCE),
            hook_source=HOOK_SOURCE,
            hook_source_sha256=digest(HOOK_SOURCE),
            browser_source=BROWSER_SOURCE,
            browser_source_sha256=digest(BROWSER_SOURCE),
            browser_image_reference="sha256:dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa",
            browser_image_content_digest="sha256:dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa",
            probe_image_reference="sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409",
            probe_image_content_digest="sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409",
            docker_network_id="79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8",
            docker_network_subnet="172.20.0.0/16",
            docker_network_gateway="172.20.0.1",
            service_unit="ullm-openai.service",
            service_user="homelab1",
            boot_id="5468046a2044427ba01b092c5ea5db6b",
            control_group="/system.slice/ullm-openai.service",
            normal_gateway_pid=1421942,
            normal_gateway_starttime_ticks=1,
            normal_worker_pid=1422312,
            normal_worker_starttime_ticks=101412600,
            normal_restart_count=1,
            restart_gateway_pid=1452201,
            restart_gateway_starttime_ticks=2,
            restart_worker_pid=1452625,
            restart_worker_starttime_ticks=101454269,
            restart_restart_count=2,
            uid=os.getuid(),
            gid=os.getgid(),
            forbidden_values=(b"formal-pilot-secret-sentinel",),
        )
        result = INGEST.ingest_failure_gate_bundle(PILOT, bindings)
        self.assertEqual(len(result.lifecycle_claims), 10)
        self.assertEqual(
            result.screenshot_evidence.sha256,
            "f5cfac6cd9b85bca472c09b088918b8a4aae525d57421724f789377a35d6fc0f",
        )
        self.assertEqual(
            result.derived_view["summary_sha256"],
            "920486a8f209710ff2bfb10113cc09120ccbf0ee4e506f842d5755aef815bc35",
        )


if __name__ == "__main__":
    unittest.main()
