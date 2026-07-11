from __future__ import annotations

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


def dataclass_replace(value, **changes):
    values = {
        field.name: getattr(value, field.name)
        for field in value.__dataclass_fields__.values()
    }
    values.update(changes)
    return type(value)(**values)


if __name__ == "__main__":
    unittest.main()
