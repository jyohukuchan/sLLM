import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_openwebui_campaign.py"
BOOT_ID = "5" * 32
NORMAL_GATEWAY_PID = 1200
NORMAL_WORKER_PID = 1201
RESTART_GATEWAY_PID = 2200
RESTART_WORKER_PID = 2201


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_openwebui_campaign", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load_module()


def compact(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def lifecycle(
    name,
    *,
    request_id="req-1",
    completion_id="chatcmpl-1",
    observed_ns=1_000_000,
):
    common = {
        "schema_version": CAMPAIGN.LIFECYCLE_SCHEMA,
        "event": name,
        "observed_monotonic_ns": observed_ns,
        "request_id": request_id,
        "completion_id": completion_id,
    }
    if name == "request_admitted":
        return {
            **common,
            "stream": True,
            "prompt_tokens": 32,
            "max_completion_tokens": 2,
        }
    if name == "request_started":
        return {
            **common,
            "stream": True,
            "prompt_tokens": 32,
            "admit_to_start_ns": 10,
        }
    if name == "request_first_token":
        return {**common, "stream": True, "completion_tokens": 1}
    if name == "request_released":
        return {
            **common,
            "stream": True,
            "outcome": "length",
            "cancel_reason": None,
            "prompt_tokens": 32,
            "completion_tokens": 2,
            "reset_complete": True,
            "admit_to_start_ns": 10,
            "start_to_release_ns": 90,
            "admit_to_release_ns": 100,
        }
    if name == "worker_fatal":
        return {**common, "reason": "worker exited", "admit_to_fatal_ns": 100}
    raise AssertionError(f"unsupported test lifecycle {name}")


def journal_line(
    cursor,
    monotonic_usec,
    pid,
    message,
    *,
    boot_id=BOOT_ID,
    unit=CAMPAIGN.SERVICE_UNIT,
    extra=None,
):
    value = {}
    if extra is not None:
        value.update(extra)
    value.update(
        {
            "__CURSOR": cursor,
            "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
            "_BOOT_ID": boot_id,
            "_PID": str(pid),
            "_SYSTEMD_UNIT": unit,
            "PRIORITY": "6",
            "MESSAGE": message,
        }
    )
    return compact(value)


def event_line(cursor, monotonic_usec, pid, event, *, prefix=True, extra=None):
    payload = compact(event).decode("utf-8")
    message = ("INFO:     " if prefix else "") + payload
    return journal_line(cursor, monotonic_usec, pid, message, extra=extra)


def trace_lines(prefix, start_usec, pid, request_id, completion_id):
    names = (
        "request_admitted",
        "request_started",
        "request_first_token",
        "request_released",
    )
    result = []
    for offset, name in enumerate(names):
        usec = start_usec + offset
        result.append(
            event_line(
                f"{prefix}-{offset}",
                usec,
                pid,
                lifecycle(
                    name,
                    request_id=request_id,
                    completion_id=completion_id,
                    observed_ns=usec * 1000 - 1,
                ),
            )
        )
    return result


class FakeJournalSource:
    GAP = object()

    def __init__(self):
        self.condition = threading.Condition()
        self.queue = deque()
        self.opened = False
        self.closed = False
        self.owner_thread = None
        self.hold = False
        self.waiting_while_held = threading.Event()
        self.release_hold = threading.Event()

    def open_after(self, unit, boot_id):
        with self.condition:
            if unit != CAMPAIGN.SERVICE_UNIT or boot_id != BOOT_ID:
                raise AssertionError("source identity differs")
            self.opened = True
            self.owner_thread = threading.get_ident()
            return "anchor-cursor"

    def read_next(self, timeout_usec):
        if threading.get_ident() != self.owner_thread:
            raise AssertionError("journal source escaped its reader thread")
        if self.hold:
            self.waiting_while_held.set()
            self.release_hold.wait(2.0)
        deadline = time.monotonic() + timeout_usec / 1_000_000
        with self.condition:
            while not self.queue and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.closed:
                return None
            item = self.queue.popleft()
        if item is self.GAP:
            raise CAMPAIGN.JournalSourceGap("fake source gap")
        return item

    def close(self):
        if self.owner_thread is not None and threading.get_ident() != self.owner_thread:
            raise AssertionError("journal source close escaped its reader thread")
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.release_hold.set()

    def feed(self, *raw_lines):
        with self.condition:
            if self.closed:
                raise RuntimeError("fake source is closed")
            self.queue.extend(raw_lines)
            self.condition.notify_all()

    def invalidate(self):
        self.feed(self.GAP)

    def hold_next_read(self):
        self.waiting_while_held.clear()
        self.release_hold.clear()
        self.hold = True
        self.condition.acquire()
        self.condition.notify_all()
        self.condition.release()
        if not self.waiting_while_held.wait(2.0):
            raise AssertionError("reader did not enter held read")

    def release_next_read(self):
        self.hold = False
        self.release_hold.set()


class CampaignJournalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_capture(self, source=None, scan_raw=None, **bounds):
        actual_source = source if source is not None else FakeJournalSource()
        actual_scan = scan_raw if scan_raw is not None else lambda _raw, _label: None
        capture = CAMPAIGN.CampaignJournalCapture(
            self.root / "service-journal.raw.jsonl",
            BOOT_ID,
            CAMPAIGN.PidEpoch(NORMAL_GATEWAY_PID, NORMAL_WORKER_PID),
            scan_raw=actual_scan,
            source=actual_source,
            **bounds,
        )
        self.addCleanup(capture.abort)
        self.assertEqual(capture.start(), "anchor-cursor")
        self.assertTrue(capture.incomplete_path.is_file())
        self.assertFalse(capture.final_path.exists())
        return capture, actual_source

    def claim_trace(self, capture, source, lines, phase, case_id, completion_id):
        source.feed(*lines)
        return capture.claim_completion_trace(
            completion_id,
            phase,
            case_id,
            time.monotonic_ns() + 2_000_000_000,
        )

    def prepare_sealable_capture(self):
        capture, source = self.make_capture()
        normal = trace_lines(
            "normal", 1000, NORMAL_GATEWAY_PID, "req-normal", "chatcmpl-normal"
        )
        normal_claimed = self.claim_trace(
            capture,
            source,
            normal,
            "resource_normal",
            "normal-001",
            "chatcmpl-normal",
        )
        self.assertEqual(len(normal_claimed), 4)
        capture.checkpoint("resource_normal", time.monotonic_ns() + 2_000_000_000)

        capture.arm_restart_transition()
        failure = trace_lines(
            "failure", 1100, NORMAL_GATEWAY_PID, "req-failure", "chatcmpl-failure"
        )[:3]
        fatal = event_line(
            "failure-3",
            1103,
            NORMAL_GATEWAY_PID,
            lifecycle(
                "worker_fatal",
                request_id="req-failure",
                completion_id="chatcmpl-failure",
                observed_ns=1_103_000 - 1,
            ),
        )
        recovery = trace_lines(
            "recovery", 1200, RESTART_GATEWAY_PID, "req-recovery", "chatcmpl-recovery"
        )
        all_failure = [*failure, fatal, *recovery]
        source.feed(*all_failure)
        claims = [
            CAMPAIGN.BundleLifecycleClaim(raw, "post_header_failure", "failure")
            for raw in failure + [fatal]
        ] + [
            CAMPAIGN.BundleLifecycleClaim(raw, "post_header_failure", "recovery")
            for raw in recovery
        ]
        claimed = capture.claim_bundle_records(
            claims, time.monotonic_ns() + 2_000_000_000
        )
        self.assertEqual(len(claimed), 8)
        capture.confirm_restart_epoch(
            CAMPAIGN.PidEpoch(RESTART_GATEWAY_PID, RESTART_WORKER_PID)
        )
        capture.checkpoint("post_header_failure", time.monotonic_ns() + 2_000_000_000)

        restart = trace_lines(
            "restart", 1300, RESTART_GATEWAY_PID, "req-restart", "chatcmpl-restart"
        )
        self.claim_trace(
            capture,
            source,
            restart,
            "resource_restart",
            "restart-001",
            "chatcmpl-restart",
        )
        capture.checkpoint("resource_restart", time.monotonic_ns() + 2_000_000_000)
        final_raw = journal_line(
            "final-cursor", 1400, RESTART_GATEWAY_PID, "final readiness"
        )
        source.feed(final_raw)
        capture.checkpoint("final", time.monotonic_ns() + 2_000_000_000)
        return capture, source, [*normal, *all_failure, *restart, final_raw]

    def test_streams_claims_and_atomically_seals_a_two_epoch_campaign(self):
        capture, _source, expected_raw = self.prepare_sealable_capture()
        self.assertEqual(
            capture.incomplete_path.read_bytes(),
            b"".join(line + b"\n" for line in expected_raw),
        )
        final_cursor = capture.seal("final-cursor", time.monotonic_ns() + 2_000_000_000)
        self.assertEqual(final_cursor, "final-cursor")
        self.assertFalse(capture.incomplete_path.exists())
        self.assertEqual(
            capture.final_path.read_bytes(),
            b"".join(line + b"\n" for line in expected_raw),
        )

    def test_completion_trace_claim_waits_for_a_racing_terminal(self):
        capture, source = self.make_capture()
        lines = trace_lines(
            "race", 1000, NORMAL_GATEWAY_PID, "req-race", "chatcmpl-race"
        )
        outcome = {}

        def claim():
            try:
                outcome["value"] = capture.claim_completion_trace(
                    "chatcmpl-race",
                    "resource_normal",
                    "race",
                    time.monotonic_ns() + 2_000_000_000,
                )
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=claim)
        thread.start()
        source.feed(*lines[:2])
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        source.feed(*lines[2:])
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        records = [item.session_hook_record() for item in outcome["value"]]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["record_type"], "gateway_event")
        self.assertEqual(records[0]["phase"], "resource_normal")
        self.assertEqual(records[0]["fields"]["journal_cursor"], "race-0")

    def test_checkpoint_drains_a_racing_row_and_rejects_it_unclaimed(self):
        capture, source = self.make_capture()
        source.feed(journal_line("ordinary", 900, NORMAL_GATEWAY_PID, "ordinary"))
        capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)
        source.hold_next_read()
        late = event_line(
            "late",
            1000,
            NORMAL_GATEWAY_PID,
            lifecycle("request_admitted", observed_ns=999_999),
        )
        source.feed(late)
        outcome = {}

        def checkpoint():
            try:
                capture.checkpoint("api_contract", time.monotonic_ns() + 2_000_000_000)
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=checkpoint)
        thread.start()
        source.release_next_read()
        thread.join(2.0)
        self.assertIsInstance(outcome.get("error"), CAMPAIGN.CampaignJournalError)
        self.assertIn("unclaimed", str(outcome["error"]))

    def test_duplicate_cursor_fails_and_abort_never_publishes(self):
        capture, source = self.make_capture()
        raw = journal_line("duplicate", 1000, NORMAL_GATEWAY_PID, "ordinary")
        source.feed(raw, raw)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "duplicated"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)
        capture.abort()
        self.assertFalse(capture.final_path.exists())
        self.assertFalse(capture.incomplete_path.exists())

    def test_source_invalidation_is_a_continuity_gap(self):
        capture, source = self.make_capture()
        source.invalidate()
        with self.assertRaisesRegex(CAMPAIGN.JournalSourceGap, "gap"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)

    def test_bundle_claim_rejects_wrong_raw_for_the_same_cursor(self):
        capture, source = self.make_capture()
        actual = event_line(
            "cursor-1",
            1000,
            NORMAL_GATEWAY_PID,
            lifecycle("request_admitted", observed_ns=999_999),
        )
        changed = event_line(
            "cursor-1",
            1000,
            NORMAL_GATEWAY_PID,
            lifecycle(
                "request_admitted",
                completion_id="chatcmpl-changed",
                observed_ns=999_999,
            ),
        )
        source.feed(actual)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "raw bytes differ"):
            capture.claim_bundle_records(
                [CAMPAIGN.BundleLifecycleClaim(changed, "openwebui", "browser-smoke")],
                time.monotonic_ns() + 2_000_000_000,
            )

    def test_bundle_claim_accepts_metadata_and_key_order_differences_only(self):
        capture, source = self.make_capture()
        event = lifecycle("request_admitted", observed_ns=999_999)
        captured = event_line(
            "cursor-1",
            1000,
            NORMAL_GATEWAY_PID,
            event,
            extra={"_HOSTNAME": "test-host"},
        )
        bundle = event_line(
            "cursor-1", 1000, NORMAL_GATEWAY_PID, event, extra={"_COMM": "gateway"}
        )
        source.feed(captured)
        claimed = capture.claim_bundle_records(
            [CAMPAIGN.BundleLifecycleClaim(bundle, "openwebui", "browser-smoke")],
            time.monotonic_ns() + 2_000_000_000,
        )
        self.assertEqual(claimed[0].fields["message"], json.loads(captured)["MESSAGE"])

    def test_wrong_gateway_pid_fails_before_restart_is_armed(self):
        capture, source = self.make_capture()
        source.feed(
            event_line(
                "wrong-pid",
                1000,
                RESTART_GATEWAY_PID,
                lifecycle("request_admitted", observed_ns=999_999),
            )
        )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "outside"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)

    def test_old_or_third_pid_after_switch_is_rejected(self):
        capture, source = self.make_capture()
        capture.checkpoint("resource_normal", time.monotonic_ns() + 2_000_000_000)
        capture.arm_restart_transition()
        source.feed(
            event_line(
                "restart-first",
                1000,
                RESTART_GATEWAY_PID,
                lifecycle("request_admitted", observed_ns=999_999),
            ),
            event_line(
                "old-after-restart",
                1001,
                NORMAL_GATEWAY_PID,
                lifecycle("request_started", observed_ns=1_000_999),
            ),
        )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "after the sole"):
            capture.checkpoint(
                "post_header_failure", time.monotonic_ns() + 2_000_000_000
            )

    def test_restart_probe_must_match_the_discovered_gateway_and_new_worker(self):
        capture, source = self.make_capture()
        capture.checkpoint("resource_normal", time.monotonic_ns() + 2_000_000_000)
        capture.arm_restart_transition()
        source.feed(
            event_line(
                "restart-first",
                1000,
                RESTART_GATEWAY_PID,
                lifecycle("request_admitted", observed_ns=999_999),
            )
        )
        deadline = time.monotonic() + 2.0
        while (
            capture.discovered_restart_gateway_pid is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "differs"):
            capture.confirm_restart_epoch(
                CAMPAIGN.PidEpoch(RESTART_GATEWAY_PID + 1, RESTART_WORKER_PID)
            )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "both change"):
            capture.confirm_restart_epoch(
                CAMPAIGN.PidEpoch(RESTART_GATEWAY_PID, NORMAL_WORKER_PID)
            )

    def test_checkpoint_rejects_unclaimed_lifecycle(self):
        capture, source = self.make_capture()
        source.feed(
            event_line(
                "unclaimed",
                1000,
                NORMAL_GATEWAY_PID,
                lifecycle("request_admitted", observed_ns=999_999),
            )
        )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "unclaimed"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)

    def test_quiet_window_does_not_advance_phase_and_rejects_racing_lifecycle(self):
        capture, source = self.make_capture()
        start_cursor = capture.wait_quiet(time.monotonic_ns() + 60_000_000)
        self.assertEqual(start_cursor, "anchor-cursor")
        capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)

        source.feed(
            event_line(
                "quiet-race",
                1000,
                NORMAL_GATEWAY_PID,
                lifecycle("request_admitted", observed_ns=999_999),
            )
        )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "quiet window"):
            capture.wait_quiet(time.monotonic_ns() + 2_000_000_000)

    def test_pending_queue_event_and_byte_bounds_fail_closed(self):
        for bound_name, bound_value in (
            ("max_pending_events", 1),
            ("max_pending_bytes", 1),
        ):
            with self.subTest(bound_name=bound_name):
                source = FakeJournalSource()
                capture, _ = self.make_capture(source, **{bound_name: bound_value})
                lines = trace_lines(
                    f"overflow-{bound_name}",
                    1000,
                    NORMAL_GATEWAY_PID,
                    "req-overflow",
                    "chatcmpl-overflow",
                )
                source.feed(*lines[:2])
                with self.assertRaisesRegex(
                    CAMPAIGN.CampaignJournalError, "pending queue overflowed"
                ):
                    capture.checkpoint(
                        "resource_normal", time.monotonic_ns() + 2_000_000_000
                    )
                capture.abort()

    def test_default_queue_holds_a_960_event_latency_bundle(self):
        capture, source = self.make_capture()
        capture.checkpoint("resource_normal", time.monotonic_ns() + 2_000_000_000)
        capture.arm_restart_transition()
        lines = []
        claims = []
        for request_index in range(240):
            case_id = f"latency-{request_index:03d}"
            trace = trace_lines(
                case_id,
                1000 + request_index * 4,
                RESTART_GATEWAY_PID,
                f"req-{request_index:03d}",
                f"chatcmpl-{request_index:03d}",
            )
            lines.extend(trace)
            claims.extend(
                CAMPAIGN.BundleLifecycleClaim(raw, "latency", case_id) for raw in trace
            )
        source.feed(*lines)
        final_cursor = "latency-239-3"
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                if capture.last_cursor == final_cursor:
                    break
            except CAMPAIGN.CampaignJournalError:
                pass
            time.sleep(0.005)
        self.assertEqual(capture.last_cursor, final_cursor)
        capture.confirm_restart_epoch(
            CAMPAIGN.PidEpoch(RESTART_GATEWAY_PID, RESTART_WORKER_PID)
        )
        claimed = capture.claim_bundle_records(
            claims, time.monotonic_ns() + 3_000_000_000
        )
        self.assertEqual(len(claimed), 960)
        capture.checkpoint("latency", time.monotonic_ns() + 2_000_000_000)

    def test_boot_unit_monotonic_and_numeric_fields_are_strict(self):
        cases = {
            "boot ID": journal_line(
                "bad-boot", 1000, NORMAL_GATEWAY_PID, "ordinary", boot_id="6" * 32
            ),
            "systemd unit": journal_line(
                "bad-unit", 1000, NORMAL_GATEWAY_PID, "ordinary", unit="other.service"
            ),
            "decimal": compact(
                {
                    "__CURSOR": "bad-pid",
                    "__MONOTONIC_TIMESTAMP": "1000",
                    "_BOOT_ID": BOOT_ID,
                    "_PID": NORMAL_GATEWAY_PID,
                    "_SYSTEMD_UNIT": CAMPAIGN.SERVICE_UNIT,
                    "PRIORITY": "6",
                    "MESSAGE": "ordinary",
                }
            ),
        }
        for expected, raw in cases.items():
            with self.subTest(expected=expected):
                capture, source = self.make_capture()
                source.feed(raw)
                with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, expected):
                    capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)
                capture.abort()

        capture, source = self.make_capture()
        source.feed(
            journal_line("later", 1001, NORMAL_GATEWAY_PID, "ordinary"),
            journal_line("earlier", 1000, NORMAL_GATEWAY_PID, "ordinary"),
        )
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "regressed"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)

    def test_row_arriving_after_final_seal_begins_is_rejected(self):
        capture, source, _raw = self.prepare_sealable_capture()
        source.hold_next_read()
        outcome = {}

        def seal():
            try:
                outcome["cursor"] = capture.seal(
                    "final-cursor", time.monotonic_ns() + 2_000_000_000
                )
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=seal)
        thread.start()
        time.sleep(0.02)
        source.feed(
            journal_line("post-seal", 1500, RESTART_GATEWAY_PID, "late ordinary row")
        )
        source.release_next_read()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), CAMPAIGN.CampaignJournalError)
        self.assertIn("after final seal began", str(outcome["error"]))
        self.assertFalse(capture.final_path.exists())
        capture.abort()
        self.assertFalse(capture.incomplete_path.exists())

    def test_zero_row_checkpoint_uses_the_campaign_start_cursor(self):
        capture, _source = self.make_capture()
        self.assertEqual(
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000),
            "anchor-cursor",
        )

    def test_raw_scanner_sees_non_lifecycle_rows_before_they_are_written(self):
        scanned = []

        def scan(raw, label):
            scanned.append((raw, label))
            if b"forbidden-secret" in raw:
                raise CAMPAIGN.CampaignJournalError("secret in journal")

        capture, source = self.make_capture(scan_raw=scan)
        safe = journal_line("safe", 1000, NORMAL_GATEWAY_PID, "ordinary")
        secret = journal_line("secret", 1001, NORMAL_GATEWAY_PID, "forbidden-secret")
        source.feed(safe, secret)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "secret"):
            capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)
        self.assertEqual(
            scanned,
            [(safe, "service journal evidence"), (secret, "service journal evidence")],
        )
        self.assertEqual(capture.incomplete_path.read_bytes(), safe + b"\n")
        self.assertFalse(capture.final_path.exists())

    def test_seal_rejects_a_non_lifecycle_row_after_run_end_cursor(self):
        capture, source, _raw = self.prepare_sealable_capture()
        run_end_cursor = "final-cursor"
        late = journal_line(
            "after-run-end", 1500, RESTART_GATEWAY_PID, "late access log"
        )
        source.feed(late)
        deadline = time.monotonic() + 2.0
        while capture.last_cursor != "after-run-end" and time.monotonic() < deadline:
            time.sleep(0.005)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "advanced"):
            capture.seal(run_end_cursor, time.monotonic_ns() + 2_000_000_000)
        self.assertFalse(capture.final_path.exists())

    def test_exclusive_publication_refuses_a_racing_final_path(self):
        capture, _source, _raw = self.prepare_sealable_capture()
        competitor = b"not campaign evidence\n"
        capture.final_path.write_bytes(competitor)
        with self.assertRaisesRegex(CAMPAIGN.CampaignJournalError, "publish"):
            capture.seal("final-cursor", time.monotonic_ns() + 2_000_000_000)
        self.assertEqual(capture.final_path.read_bytes(), competitor)
        self.assertFalse(capture.incomplete_path.exists())

    def test_explicit_abort_removes_incomplete_and_never_publishes(self):
        capture, source = self.make_capture()
        source.feed(journal_line("ordinary", 1000, NORMAL_GATEWAY_PID, "ordinary"))
        capture.checkpoint("preflight", time.monotonic_ns() + 2_000_000_000)
        capture.abort()
        self.assertTrue(source.closed)
        self.assertFalse(capture.incomplete_path.exists())
        self.assertFalse(capture.final_path.exists())


if __name__ == "__main__":
    unittest.main()
