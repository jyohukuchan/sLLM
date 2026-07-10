from __future__ import annotations

import importlib.util
import fcntl
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "run-sq8-worker-acceptance.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("run_sq8_worker_acceptance", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def compact(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def ready_event():
    return {
        "schema_version": TOOL.WORKER_SCHEMA,
        "type": "ready",
        "model": "ullm-qwen3-14b-sq8",
        "model_revision": TOOL.EXPECTED_MODEL_REVISION,
        "artifact_content_sha256": TOOL.EXPECTED_ARTIFACT_CONTENT_SHA256,
        "package_manifest_sha256": TOOL.EXPECTED_PACKAGE_MANIFEST_SHA256,
        "device": "gfx1201",
        "execution_profile": "rdna4_w8a8_block_ck",
        "context_length": 4096,
        "max_new_tokens": 512,
    }


def event(kind: str, request_id: str = "req-1", **fields):
    value = {"schema_version": TOOL.WORKER_SCHEMA, "type": kind, "request_id": request_id}
    value.update(fields)
    return value


def proc_stat(pid: int, ppid: int, starttime: int, comm: str = "worker ) name") -> str:
    remaining = [str(field) for field in range(4, 53)]
    remaining[0] = str(ppid)
    remaining[18] = str(starttime)
    return f"{pid} ({comm}) S " + " ".join(remaining) + "\n"


def make_proc_tree(root: Path, pid: int, ppid: int = 10, starttime: int = 1234):
    process = root / str(pid)
    (process / "fd").mkdir(parents=True)
    (process / "task" / str(pid)).mkdir(parents=True)
    (process / "stat").write_text(proc_stat(pid, ppid, starttime), encoding="ascii")
    (process / "status").write_text(
        "Name:\tworker\nVmRSS:\t  12345 kB\nThreads:\t7\n", encoding="ascii"
    )
    (process / "task" / str(pid) / "children").write_text("101 102\n", encoding="ascii")
    (process / "fd" / "0").write_text("", encoding="ascii")
    (process / "fd" / "1").write_text("", encoding="ascii")
    executable = root / "ullm-sq8-worker"
    executable.write_text("binary", encoding="ascii")
    os.symlink(executable, process / "exe")
    return executable


def amd_list_raw():
    return compact(
        [
            {
                "gpu": TOOL.GPU_INDEX,
                "bdf": TOOL.GPU_BDF,
                "uuid": TOOL.GPU_UUID,
                "kfd_id": TOOL.KFD_GPU_ID,
            }
        ]
    )


def amd_process_raw(pid: int, vram: int = 20_000_000_000):
    return compact(
        [
            {
                "gpu": TOOL.GPU_INDEX,
                "process_list": [
                    {
                        "process_info": {
                            "pid": pid,
                            "mem_usage": {"value": vram, "unit": "B"},
                        }
                    }
                ],
            }
        ]
    )


class CaptureEvidence:
    def __init__(self):
        self.records = []

    def write(self, value):
        self.records.append(value)


class FakeTransport:
    def __init__(self):
        self.generated = []
        self.cancelled = []

    def generate(self, spec):
        self.generated.append(spec)
        return time.monotonic_ns()

    def cancel(self, spec):
        self.cancelled.append(spec)
        return time.monotonic_ns()


class FakeEvents:
    def __init__(self, values):
        self.values = list(values)

    def next(self, _deadline):
        if not self.values:
            raise AssertionError("event fixture exhausted")
        value = self.values.pop(0)
        raw = compact(value)
        return TOOL.PumpEvent(
            time.monotonic_ns(), value, raw.decode("utf-8"), TOOL.sha256_bytes(raw)
        )


class StrictJsonAndPumpTests(unittest.TestCase):
    def test_strict_json_rejects_duplicates_nonfinite_utf8_and_oversize(self):
        self.assertEqual(TOOL.strict_json_object(b'{"a":1}', "fixture"), {"a": 1})
        for raw, message in (
            (b'{"a":1,"a":2}', "duplicate"),
            (b'{"a":NaN}', "non-finite"),
            (b'{"a":1e999}', "non-finite"),
            (b'{"a":"\xff"}', "UTF-8"),
        ):
            with self.assertRaisesRegex(TOOL.AcceptanceError, message):
                TOOL.strict_json_bytes(raw, "fixture")
        with self.assertRaisesRegex(TOOL.AcceptanceError, "8 MiB"):
            TOOL.strict_json_bytes(b" " * (TOOL.MAX_LINE_BYTES + 1), "fixture")

    def test_released_reset_complete_rejects_integer_one(self):
        released = event(
            "released",
            outcome="length",
            prompt_tokens=8,
            completion_tokens=2,
            reset_complete=1,
        )
        with self.assertRaisesRegex(TOOL.AcceptanceError, "must be true"):
            TOOL.validate_worker_event_shape(released)

    def test_stdout_pump_decodes_before_timestamp_and_reports_eof(self):
        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "rb", buffering=0)
        writer = os.fdopen(write_fd, "wb", buffering=0)
        pump = TOOL.StdoutPump(reader)
        pump.start()
        before = time.monotonic_ns()
        writer.write(compact(ready_event()) + b"\n")
        writer.close()
        item = pump.receive(time.monotonic_ns() + 1_000_000_000)
        self.assertIsInstance(item, TOOL.PumpEvent)
        self.assertGreaterEqual(item.observed_monotonic_ns, before)
        self.assertEqual(item.event, ready_event())
        self.assertEqual(item.raw_json, compact(ready_event()).decode("utf-8"))
        self.assertEqual(item.raw_sha256, TOOL.sha256_bytes(compact(ready_event())))
        eof = pump.receive(time.monotonic_ns() + 1_000_000_000)
        self.assertIsInstance(eof, TOOL.PumpEof)
        pump.close()

    def test_stdout_pump_rejects_partial_and_oversized_lines(self):
        for raw, message in (
            (b'{"schema_version":"ullm.worker.v1"}', "partial"),
            (b"x" * (TOOL.MAX_LINE_BYTES + 1), "8 MiB"),
        ):
            with self.subTest(message=message):
                pump = TOOL.StdoutPump(io.BytesIO(raw))
                pump.start()
                with self.assertRaisesRegex(TOOL.AcceptanceError, message):
                    pump.receive(time.monotonic_ns() + 1_000_000_000)
                pump.close()

    def test_stderr_drain_preserves_exact_bytes(self):
        read_fd, write_fd = os.pipe()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stderr.incomplete"
            output = path.open("wb", buffering=0)
            drain = TOOL.StderrDrain(os.fdopen(read_fd, "rb", buffering=0), output)
            drain.start()
            raw = b'{"event":"one"}\n\x00tail\n'
            with os.fdopen(write_fd, "wb", buffering=0) as writer:
                writer.write(raw)
            drain.join()
            self.assertEqual(path.read_bytes(), raw)

    def test_worker_event_record_preserves_exact_stdout_json_and_hash(self):
        value = ready_event()
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True)

        class Pump:
            def receive(self, _deadline):
                return TOOL.PumpEvent(
                    123,
                    value,
                    raw,
                    TOOL.sha256_bytes(raw.encode("utf-8")),
                )

        evidence = CaptureEvidence()
        item = TOOL.WorkerEvents(Pump(), evidence).next(999)
        self.assertEqual(item.event, value)
        self.assertEqual(evidence.records[0]["raw_json"], raw)
        self.assertEqual(
            evidence.records[0]["raw_sha256"],
            TOOL.sha256_bytes(raw.encode("utf-8")),
        )


class EventStateMachineTests(unittest.TestCase):
    def test_normal_request_requires_progress_contiguous_tokens_and_length(self):
        spec = TOOL.normal_spec("resource_measured", 1, "req-1")
        events = FakeEvents(
            [
                event("started", prompt_tokens=8),
                event("progress", phase="prefill", processed_prompt_tokens=8),
                event("token", index=0, token_id=353),
                event("token", index=1, token_id=10),
                event(
                    "released",
                    outcome="length",
                    prompt_tokens=8,
                    completion_tokens=2,
                    reset_complete=True,
                ),
            ]
        )
        transport = FakeTransport()
        release = TOOL.run_request(spec, transport, events)
        self.assertEqual((release.outcome, release.completion_tokens), ("length", 2))
        self.assertEqual(transport.cancelled, [])

    def test_prompt_and_decode_cancel_at_the_required_boundaries(self):
        prompt_spec = TOOL.cancel_spec("latency_warmup", 1, "prompt-1", "prompt")
        prompt_transport = FakeTransport()
        prompt_release = TOOL.run_request(
            prompt_spec,
            prompt_transport,
            FakeEvents(
                [
                    event("started", "prompt-1", prompt_tokens=128),
                    event(
                        "released",
                        "prompt-1",
                        outcome="cancelled",
                        cancel_reason="operator",
                        prompt_tokens=128,
                        completion_tokens=0,
                        reset_complete=True,
                    ),
                ]
            ),
        )
        self.assertEqual(prompt_transport.cancelled, [prompt_spec])
        self.assertEqual(prompt_release.completion_tokens, 0)

        decode_spec = TOOL.cancel_spec("latency_warmup", 2, "decode-2", "decode")
        decode_transport = FakeTransport()
        decode_release = TOOL.run_request(
            decode_spec,
            decode_transport,
            FakeEvents(
                [
                    event("started", "decode-2", prompt_tokens=8),
                    event("progress", "decode-2", phase="prefill", processed_prompt_tokens=8),
                    event("token", "decode-2", index=0, token_id=353),
                    event(
                        "released",
                        "decode-2",
                        outcome="cancelled",
                        cancel_reason="operator",
                        prompt_tokens=8,
                        completion_tokens=1,
                        reset_complete=True,
                    ),
                ]
            ),
        )
        self.assertEqual(decode_transport.cancelled, [decode_spec])
        self.assertEqual(decode_release.completion_tokens, 1)

    def test_cancel_target_to_write_gap_rejects_more_than_thirty_seconds(self):
        spec = TOOL.cancel_spec("latency_warmup", 1, "prompt-1", "prompt")
        transport = mock.Mock()
        observed = 100
        transport.cancel.return_value = observed + TOOL.PROGRESS_TIMEOUT_NS + 1
        with self.assertRaisesRegex(TOOL.AcceptanceError, "gap exceeds"):
            TOOL.cancel_after_target(spec, transport, observed)

    def test_state_machine_rejects_prompt_cancel_token_and_noncontiguous_token(self):
        prompt = TOOL.cancel_spec("latency_warmup", 1, "req-prompt", "prompt")
        with self.assertRaisesRegex(TOOL.AcceptanceError, "prompt-target"):
            TOOL.run_request(
                prompt,
                FakeTransport(),
                FakeEvents(
                    [
                        event("started", "req-prompt", prompt_tokens=128),
                        event("progress", "req-prompt", phase="prefill", processed_prompt_tokens=128),
                        event("token", "req-prompt", index=0, token_id=1),
                    ]
                ),
            )
        normal = TOOL.normal_spec("resource_measured", 1, "req-normal")
        with self.assertRaisesRegex(TOOL.AcceptanceError, "index"):
            TOOL.run_request(
                normal,
                FakeTransport(),
                FakeEvents(
                    [
                        event("started", "req-normal", prompt_tokens=8),
                        event("progress", "req-normal", phase="prefill", processed_prompt_tokens=8),
                        event("token", "req-normal", index=1, token_id=1),
                    ]
                ),
            )

    def test_schedule_ids_targets_and_latency_recovery_contract(self):
        expected_cancel_targets = []
        for index in range(1, 101):
            spec = TOOL.resource_request_spec("resource_measured", index)
            self.assertEqual(spec.request_id, f"p8c-resource-measured-{index:03d}")
            if index % 5 == 4:
                expected_cancel_targets.append(spec.cancel_target)
            else:
                self.assertIsNone(spec.cancel_target)
        self.assertEqual(expected_cancel_targets, ["prompt", "decode"] * 10)
        cancelled = "p8c-latency-measured-01"
        recovery = TOOL.normal_spec("latency_measured", 1, cancelled + "-recovery")
        self.assertEqual(recovery.request_id, "p8c-latency-measured-01-recovery")
        self.assertEqual((recovery.prompt_tokens, recovery.max_new_tokens), (8, 2))


class CommandAndEvidenceTests(unittest.TestCase):
    def test_progress_is_written_to_stderr_with_flush(self):
        with mock.patch("builtins.print") as output:
            TOOL.emit_progress("phase latency_warmup started")
        output.assert_called_once_with(
            "[sq8-acceptance] phase latency_warmup started",
            file=TOOL.sys.stderr,
            flush=True,
        )

    def test_commands_record_write_start_before_completion_and_fixed_hash(self):
        evidence = CaptureEvidence()
        stdin = io.BytesIO()
        transport = TOOL.WorkerTransport(stdin, evidence)
        spec = TOOL.cancel_spec("latency_measured", 1, "p8c-latency-measured-01", "prompt")
        with mock.patch.object(
            TOOL, "next_strict_timestamp", side_effect=[10, 11, 20, 21, 30, 31]
        ), mock.patch.object(TOOL.time, "monotonic_ns", return_value=31):
            self.assertEqual(transport.generate(spec), 10)
            self.assertEqual(transport.cancel(spec), 20)
            self.assertEqual(transport.shutdown(), 30)
        self.assertEqual([record["command_type"] for record in evidence.records], ["generate", "cancel", "shutdown"])
        generate = evidence.records[0]
        self.assertEqual(generate["write_started_monotonic_ns"], 10)
        self.assertEqual(generate["write_completed_monotonic_ns"], 11)
        self.assertEqual(generate["prompt_token_ids_sha256"], TOOL.prompt_sha256(128))
        self.assertEqual(
            generate["raw_sha256"],
            TOOL.sha256_bytes(generate["raw_json"].encode("utf-8")),
        )
        self.assertEqual(json.loads(generate["raw_json"])["type"], "generate")
        self.assertEqual(evidence.records[1]["cancel_target"], "prompt")
        commands = [json.loads(line) for line in stdin.getvalue().splitlines()]
        self.assertEqual(commands[0]["prompt_token_ids"], list(range(1, 129)))
        self.assertEqual(commands[-1], {"schema_version": TOOL.WORKER_SCHEMA, "type": "shutdown"})

    def test_shutdown_write_deadline_is_based_on_write_start(self):
        evidence = CaptureEvidence()
        transport = TOOL.WorkerTransport(io.BytesIO(), evidence)
        with mock.patch.object(
            TOOL, "next_strict_timestamp", side_effect=[100, 101]
        ), mock.patch.object(TOOL, "write_all_before_deadline") as write:
            self.assertEqual(transport.shutdown(), 100)
        self.assertEqual(write.call_args.args[2], 100 + TOOL.SHUTDOWN_TIMEOUT_NS)

    def test_all_command_write_deadlines_use_the_recorded_write_start(self):
        evidence = CaptureEvidence()
        transport = TOOL.WorkerTransport(io.BytesIO(), evidence)
        spec = TOOL.cancel_spec("latency_measured", 1, "cancel-1", "prompt")
        with mock.patch.object(
            TOOL,
            "next_strict_timestamp",
            side_effect=[100, 101, 200, 201, 300, 301],
        ), mock.patch.object(TOOL, "write_all_before_deadline") as write:
            self.assertEqual(transport.generate(spec), 100)
            self.assertEqual(transport.cancel(spec), 200)
            self.assertEqual(transport.shutdown(), 300)
        self.assertEqual(
            [(call.args[2], call.args[3]) for call in write.call_args_list],
            [
                (100 + TOOL.PROGRESS_TIMEOUT_NS, "worker generate write"),
                (200 + TOOL.CANCEL_TIMEOUT_NS, "worker cancel write"),
                (300 + TOOL.SHUTDOWN_TIMEOUT_NS, "worker shutdown write"),
            ],
        )

    def test_bounded_shutdown_write_rejects_nonwritable_pipe(self):
        stream = mock.Mock()
        stream.fileno.return_value = 9
        deadline = time.monotonic_ns() + 1_000_000_000
        with mock.patch.object(TOOL.os, "get_blocking", return_value=True), mock.patch.object(
            TOOL.os, "set_blocking"
        ) as set_blocking, mock.patch.object(
            TOOL.os, "write", side_effect=BlockingIOError
        ), mock.patch.object(
            TOOL.select, "select", return_value=([], [], [])
        ):
            with self.assertRaisesRegex(TOOL.AcceptanceError, "absolute deadline"):
                TOOL.write_all_before_deadline(stream, b"shutdown\n", deadline)
        self.assertEqual(
            set_blocking.call_args_list,
            [mock.call(9, False), mock.call(9, True)],
        )

    def test_bounded_writer_rejects_a_full_pipe_with_stalled_reader(self):
        read_fd, write_fd = os.pipe()
        writer = None
        try:
            try:
                fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, 4096)
            except OSError:
                pass
            os.set_blocking(write_fd, False)
            while True:
                try:
                    os.write(write_fd, b"x" * 4096)
                except BlockingIOError:
                    break
            os.set_blocking(write_fd, True)
            writer = os.fdopen(write_fd, "wb", buffering=0)
            write_fd = -1
            deadline = time.monotonic_ns() + 50_000_000
            with self.assertRaisesRegex(TOOL.AcceptanceError, "absolute deadline"):
                TOOL.write_all_before_deadline(
                    writer,
                    b'{"type":"cancel"}\n',
                    deadline,
                    "worker cancel write",
                )
        finally:
            if writer is not None:
                writer.close()
            elif write_fd >= 0:
                os.close(write_fd)
            os.close(read_fd)

    def test_evidence_is_incomplete_until_successful_atomic_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete_dir = root / "failed"
            failed = TOOL.EvidenceWriter(incomplete_dir)
            failed.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "header"})
            stderr = failed.open_stderr()
            stderr.write(b"partial stderr\n")
            stderr.close()
            failed.abort()
            self.assertTrue((incomplete_dir / "raw.jsonl.incomplete").is_file())
            self.assertFalse((incomplete_dir / "raw.jsonl").exists())
            with self.assertRaisesRegex(TOOL.AcceptanceError, "fresh evidence"):
                TOOL.EvidenceWriter(incomplete_dir)

            complete_dir = root / "complete"
            complete = TOOL.EvidenceWriter(complete_dir)
            complete.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "header"})
            stderr = complete.open_stderr()
            stderr.write(b"{}\n")
            stderr.close()
            digest = complete.finish_stderr()
            self.assertEqual(digest, TOOL.sha256_bytes(b"{}\n"))
            complete.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "process_exit"})
            complete.publish()
            self.assertTrue((complete_dir / "raw.jsonl").is_file())
            self.assertTrue((complete_dir / "worker-stderr.jsonl").is_file())
            self.assertFalse((complete_dir / "raw.jsonl.incomplete").exists())

    def test_publish_rejects_output_path_identity_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            output = parent / "evidence"
            writer = TOOL.EvidenceWriter(output)
            writer.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "header"})

            moved_parent = root / "moved-parent"
            parent.rename(moved_parent)
            parent.mkdir()
            output.mkdir()
            with self.assertRaisesRegex(TOOL.AcceptanceError, "identity changed"):
                writer.publish()
            self.assertFalse((output / "raw.jsonl").exists())
            self.assertTrue(
                (moved_parent / "evidence" / "raw.jsonl.incomplete").is_file()
            )

    def test_publish_rolls_back_final_name_when_directory_fsync_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            writer = TOOL.EvidenceWriter(output)
            writer.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "header"})
            real_fsync = TOOL.os.fsync
            calls = 0

            def fail_directory_fsync(descriptor):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic directory fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(TOOL.os, "fsync", side_effect=fail_directory_fsync):
                with self.assertRaisesRegex(TOOL.AcceptanceError, "failed to publish"):
                    writer.publish()
            self.assertFalse((output / "raw.jsonl").exists())
            self.assertTrue((output / "raw.jsonl.incomplete").is_file())

    def test_publish_rolls_back_final_name_before_reraising_interrupt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            writer = TOOL.EvidenceWriter(output)
            writer.write({"schema_version": TOOL.RAW_SCHEMA, "record_type": "header"})
            real_fsync = TOOL.os.fsync
            calls = 0

            def interrupt_directory_fsync(descriptor):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return real_fsync(descriptor)

            with mock.patch.object(
                TOOL.os, "fsync", side_effect=interrupt_directory_fsync
            ):
                with self.assertRaises(KeyboardInterrupt):
                    writer.publish()
            self.assertFalse((output / "raw.jsonl").exists())
            self.assertTrue((output / "raw.jsonl.incomplete").is_file())

    def test_frozen_statistics(self):
        values = list(range(1, 11))
        self.assertEqual(
            TOOL.percentile_linear(values, TOOL.fractions.Fraction(95, 100)),
            TOOL.fractions.Fraction(191, 20),
        )
        self.assertEqual(TOOL.median([1, 10, 3, 2, 4]), 3)
        self.assertEqual(TOOL.theil_sen([100, 110, 120, 130]), 10)

    def test_git_porcelain_allows_only_profiler_descendants_and_preserves_raw(self):
        raw = b"?? .rocprofv3/\n?? .rocprofv3/one.dat\n?? .rocprofv3/sub/two.dat\n"
        self.assertEqual(TOOL.validate_git_status_raw(raw), raw.decode("utf-8"))
        self.assertEqual(TOOL.validate_git_status_raw(b""), "")
        for invalid in (
            b" M tracked.py\n",
            b"?? other.txt\n",
            b"?? .rocprofv3/../escape\n",
        ):
            with self.subTest(raw=invalid):
                with self.assertRaises(TOOL.AcceptanceError):
                    TOOL.validate_git_status_raw(invalid)

    def test_output_directory_must_be_fresh_outside_repo_without_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            valid = root / "evidence"
            self.assertEqual(
                TOOL.prospective_output_directory(valid, repo), valid
            )
            with self.assertRaisesRegex(TOOL.AcceptanceError, "outside"):
                TOOL.prospective_output_directory(repo / "evidence", repo)

            outside = root / "outside"
            outside.mkdir()
            symlink = root / "linked-output-parent"
            symlink.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(TOOL.AcceptanceError, "symlink"):
                TOOL.prospective_output_directory(symlink / "evidence", repo)

    def test_repo_root_must_be_actual_git_worktree_toplevel(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            subprocess.run(
                ["git", "init", "--quiet", str(repo)],
                check=True,
                capture_output=True,
            )
            nested = repo / "docs"
            nested.mkdir()
            TOOL.require_git_toplevel(repo.resolve())
            with self.assertRaisesRegex(TOOL.AcceptanceError, "top-level"):
                TOOL.require_git_toplevel(nested.resolve())

    def test_output_creation_rejects_parent_symlink_swap_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            parent = root / "evidence-parent"
            parent.mkdir()
            replacement = root / "replacement"
            replacement.mkdir()
            moved_parent = root / "moved-evidence-parent"
            output = parent / "evidence"
            original = TOOL.prospective_output_directory

            def swap_parent(output_dir, repo_root):
                prospective = original(output_dir, repo_root)
                parent.rename(moved_parent)
                parent.symlink_to(replacement, target_is_directory=True)
                return prospective

            with mock.patch.object(
                TOOL, "prospective_output_directory", side_effect=swap_parent
            ):
                with self.assertRaisesRegex(TOOL.AcceptanceError, "without symlinks"):
                    TOOL.create_fresh_output_directory(output, repo)
            self.assertFalse((replacement / "evidence").exists())

    def test_final_git_identity_rejects_commit_change(self):
        identity = mock.Mock(repo_root=Path("/repo"), git_commit="a" * 40)
        with mock.patch.object(
            TOOL, "git_identity", return_value=("b" * 40, "")
        ):
            with self.assertRaisesRegex(TOOL.AcceptanceError, "HEAD changed"):
                TOOL.final_git_identity(identity)

    def test_process_group_cleanup_uses_term_then_kill_with_two_second_waits(self):
        process = mock.Mock()
        process.pid = 4242
        process.poll.side_effect = [0, 0]
        process.wait.return_value = 0
        zero_calls = 0

        def killpg(_process_group, signum):
            nonlocal zero_calls
            if signum == 0:
                zero_calls += 1
                if zero_calls == 2:
                    raise ProcessLookupError

        with mock.patch.object(TOOL.os, "killpg", side_effect=killpg) as signal_group, mock.patch.object(
            TOOL.time, "monotonic_ns", side_effect=[0, 2_000_000_000, 4_000_000_000]
        ), mock.patch.object(TOOL.time, "sleep"):
            TOOL.terminate_process_group(process, 4242)
        self.assertEqual(
            [call.args[1] for call in signal_group.call_args_list],
            [signal.SIGTERM, 0, signal.SIGKILL, 0],
        )
        process.wait.assert_called_once_with(timeout=0)

    def test_process_group_cleanup_fails_if_kill_wait_expires(self):
        process = mock.Mock()
        process.pid = 4242
        process.poll.return_value = 0
        with mock.patch.object(TOOL.os, "killpg"), mock.patch.object(
            TOOL.time,
            "monotonic_ns",
            side_effect=[0, 2_000_000_000, 4_000_000_000, 6_000_000_000],
        ), mock.patch.object(TOOL.time, "sleep"):
            with self.assertRaisesRegex(TOOL.AcceptanceError, "remained after SIGKILL"):
                TOOL.terminate_process_group(process, 4242)

    def test_process_group_cleanup_kills_descendant_after_leader_exits_on_term(self):
        script = """
import os
import signal
import time

child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(60)
else:
    time.sleep(60)
"""
        process = TOOL.subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=TOOL.subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.assertIsNotNone(process.stdout)
        descendant_pid = int(process.stdout.readline().strip())
        try:
            self.assertEqual(os.getpgid(descendant_pid), process.pid)
            TOOL.terminate_process_group(process, process.pid, grace_seconds=0.1)
            self.assertIsNotNone(process.poll())
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant_pid, 0)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)
            process.stdout.close()

    def test_external_commands_start_a_dedicated_session(self):
        captured = {}

        class Process:
            pid = 4242

            @staticmethod
            def wait(timeout):
                self.assertEqual(timeout, TOOL.COMMAND_TIMEOUT_SECONDS)
                return 0

        def spawn(arguments, **kwargs):
            captured.update(kwargs)
            kwargs["stdout"].write(b"output\n")
            return Process()

        with mock.patch.object(TOOL.subprocess, "Popen", side_effect=spawn):
            self.assertEqual(TOOL.run_bounded_command(["fake"], "fake command"), b"output\n")
        self.assertIs(captured["start_new_session"], True)

    def test_worker_starts_a_dedicated_session(self):
        identity = mock.Mock(
            worker=Path("/worker"),
            artifact=Path("/artifact"),
            package=Path("/package"),
        )
        sentinel = object()
        with mock.patch.object(TOOL.subprocess, "Popen", return_value=sentinel) as popen:
            self.assertIs(TOOL.spawn_worker(identity, {"HIP_VISIBLE_DEVICES": "1"}), sentinel)
        arguments = popen.call_args.args[0]
        options = popen.call_args.kwargs
        self.assertEqual(
            arguments,
            ["/worker", "--artifact", "/artifact", "--package", "/package"],
        )
        self.assertIs(options["start_new_session"], True)
        self.assertEqual(options["bufsize"], 0)

    def test_shutdown_remaining_time_uses_one_absolute_deadline(self):
        with mock.patch.object(TOOL.time, "monotonic_ns", return_value=2_000_000_000):
            self.assertEqual(
                TOOL.remaining_seconds(32_000_000_000, "shutdown"), 30.0
            )
        with mock.patch.object(TOOL.time, "monotonic_ns", return_value=32_000_000_001):
            with self.assertRaisesRegex(TOOL.AcceptanceError, "absolute deadline"):
                TOOL.remaining_seconds(32_000_000_000, "shutdown")


class ProcAndGpuProbeTests(unittest.TestCase):
    def test_proc_stat_uses_rightmost_valid_delimiter(self):
        ppid, starttime = TOOL.parse_proc_stat(
            proc_stat(222, 111, 987654, "strange ) S comm)"), 222
        )
        self.assertEqual((ppid, starttime), (111, 987654))
        with self.assertRaisesRegex(TOOL.AcceptanceError, "delimiter"):
            TOOL.parse_proc_stat("222 (broken stat\n", 222)

    def test_proc_capture_checks_identity_rss_threads_fds_and_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = make_proc_tree(root, 222)
            capture = TOOL.capture_worker_proc(root, 222)
            self.assertEqual(capture["ppid"], 10)
            self.assertEqual(capture["starttime_ticks_before"], 1234)
            self.assertEqual(capture["vmrss_bytes"], 12345 * 1024)
            self.assertEqual(capture["threads"], 7)
            self.assertEqual(capture["fd_count"], 2)
            self.assertEqual(capture["children"], [101, 102])
            self.assertEqual(capture["exe"], str(executable))
            self.assertEqual(capture["exe_target"], str(executable))
            self.assertEqual(capture["fd_names"], ["0", "1"])
            for raw_name, sha_name in (
                ("stat_before_raw", "stat_before_raw_sha256"),
                ("status_raw", "status_raw_sha256"),
                ("children_raw", "children_raw_sha256"),
                ("stat_after_raw", "stat_after_raw_sha256"),
            ):
                self.assertEqual(
                    capture[sha_name],
                    TOOL.sha256_bytes(capture[raw_name].encode("utf-8")),
                )

    def test_kfd_probe_records_all_positive_processes_in_pid_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, amount in ((30, 0), (20, 200), (10, 100)):
                process = root / str(pid)
                process.mkdir()
                (process / f"vram_{TOOL.KFD_GPU_ID}").write_text(f"{amount}\n", encoding="ascii")
            self.assertEqual(
                TOOL.scan_kfd_positive(root),
                [{"pid": 10, "vram_bytes": 100}, {"pid": 20, "vram_bytes": 200}],
            )
            self.assertEqual(
                TOOL.capture_kfd_processes(root),
                [
                    {"pid": 10, "vram_raw": "100", "vram_bytes": 100},
                    {"pid": 20, "vram_raw": "200", "vram_bytes": 200},
                    {"pid": 30, "vram_raw": "0", "vram_bytes": 0},
                ],
            )

    def test_kfd_probe_rejects_nonpositive_numeric_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = Path(temporary) / "0"
            process.mkdir()
            (process / f"vram_{TOOL.KFD_GPU_ID}").write_text("0\n", encoding="ascii")
            with self.assertRaisesRegex(TOOL.AcceptanceError, "PID must be positive"):
                TOOL.capture_kfd_processes(Path(temporary))

    def test_amd_smi_identity_and_process_parser(self):
        TOOL.parse_amd_smi_list(amd_list_raw())
        duplicate_index = json.loads(amd_list_raw())
        duplicate_index.append(
            {
                "gpu": TOOL.GPU_INDEX,
                "bdf": "0000:00:00.0",
                "uuid": "wrong",
                "kfd_id": 0,
            }
        )
        with self.assertRaisesRegex(TOOL.AcceptanceError, "unique matching GPU index 2"):
            TOOL.parse_amd_smi_list(compact(duplicate_index))
        self.assertEqual(TOOL.parse_amd_process(amd_process_raw(222), 222), 20_000_000_000)
        wrong = json.loads(amd_process_raw(222))
        wrong[0]["process_list"][0]["process_info"]["mem_usage"]["unit"] = "MiB"
        with self.assertRaisesRegex(TOOL.AcceptanceError, "unit"):
            TOOL.parse_amd_process(compact(wrong), 222)

    def test_gpu_metric_requires_exactly_gpu_two(self):
        raw = compact({"gpu_data": [{"gpu": TOOL.GPU_INDEX, "temperature": {}}]})
        record = TOOL.capture_gpu_metric("amd-smi", lambda _args, _label: raw, "before")
        self.assertEqual(record["boundary"], "before")
        self.assertEqual(record["raw_sha256"], TOOL.sha256_bytes(raw))
        wrong = compact({"gpu_data": [{"gpu": 1}]})
        with self.assertRaisesRegex(TOOL.AcceptanceError, "GPU 2"):
            TOOL.capture_gpu_metric("amd-smi", lambda _args, _label: wrong, "before")

    def test_resource_sample_cross_checks_proc_amd_and_kfd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_root = root / "proc"
            kfd_root = root / "kfd"
            proc_root.mkdir()
            kfd_root.mkdir()
            make_proc_tree(proc_root, 222)
            kfd_process = kfd_root / "222"
            kfd_process.mkdir()
            (kfd_process / f"vram_{TOOL.KFD_GPU_ID}").write_text(
                "20000000000\n", encoding="ascii"
            )

            def runner(arguments, label):
                self.assertEqual(label, "amd-smi process")
                self.assertIn("process", arguments)
                return amd_process_raw(222)

            context = TOOL.ProbeContext("amd-smi", proc_root, kfd_root, runner)
            captured, rss, vram = TOOL.capture_resource_sample(context, 222)
            self.assertEqual(rss, 12345 * 1024)
            self.assertEqual(vram, 20_000_000_000)
            self.assertEqual(
                captured["gpu"]["kfd_positive_processes"],
                [{"pid": 222, "vram_bytes": 20_000_000_000}],
            )
            self.assertEqual(
                captured["gpu"]["kfd_processes"],
                [
                    {
                        "pid": 222,
                        "vram_raw": "20000000000",
                        "vram_bytes": 20_000_000_000,
                    }
                ],
            )
            self.assertEqual(captured["gpu"]["unrelated_positive_kfd_pids"], [])

            other = kfd_root / "333"
            other.mkdir()
            (other / f"vram_{TOOL.KFD_GPU_ID}").write_text("1\n", encoding="ascii")
            with self.assertRaisesRegex(TOOL.AcceptanceError, "isolated KFD"):
                TOOL.capture_resource_sample(context, 222)

    def test_isolation_check_keeps_raw_zero_and_positive_kfd_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, amount in ((111, 0), (222, 500)):
                process = root / str(pid)
                process.mkdir()
                (process / f"vram_{TOOL.KFD_GPU_ID}").write_text(
                    f" {amount}\n", encoding="ascii"
                )
            evidence = CaptureEvidence()
            TOOL.record_isolation_check(
                evidence,
                root,
                222,
                phase="resource_measured",
                request_index=7,
                request_id="p8c-resource-measured-007",
                release_observed_monotonic_ns=100,
            )
            record = evidence.records[0]
            self.assertEqual(record["record_type"], "isolation_check")
            self.assertGreater(record["captured_monotonic_ns"], 100)
            self.assertEqual(
                record["kfd_processes"],
                [
                    {"pid": 111, "vram_raw": "0", "vram_bytes": 0},
                    {"pid": 222, "vram_raw": "500", "vram_bytes": 500},
                ],
            )


class HeaderAndCliTests(unittest.TestCase):
    def test_header_has_exact_frozen_nested_fields_and_raw_hashes(self):
        identity = TOOL.Preflight(
            repo_root=Path("/repo"),
            output_dir=Path("/evidence"),
            worker=Path("/worker"),
            artifact=Path("/artifact"),
            package=Path("/package"),
            git_commit="a" * 40,
            git_status_raw="?? .rocprofv3/counters.dat\n",
            binary_sha256="b" * 64,
            amd_version_raw=b"version\n",
            amd_list_raw=amd_list_raw(),
            guards={name: "1" for name in TOOL.REQUIRED_HIP_GUARDS},
            preflight_kfd_processes=[
                {"pid": 111, "vram_raw": "0", "vram_bytes": 0}
            ],
        )
        header = TOOL.header_record(identity, 222, 111, 1234, "/worker")
        TOOL.exact_keys(
            header,
            {
                "schema_version",
                "record_type",
                "clock",
                "build",
                "worker",
                "device",
                "environment",
                "schedule",
                "thresholds",
            },
            "header",
        )
        self.assertEqual(header["device"]["amd_smi_list_raw_sha256"], TOOL.sha256_bytes(amd_list_raw()))
        self.assertEqual(set(header["environment"]["required_hip_guards"]), set(TOOL.REQUIRED_HIP_GUARDS))
        self.assertEqual(header["schedule"]["resource_requests"], 100)
        self.assertEqual(header["thresholds"]["shutdown_max_ns"], 30_000_000_000)
        self.assertEqual(
            header["build"]["git_status_raw_sha256"],
            TOOL.sha256_bytes(header["build"]["git_status_raw"].encode("utf-8")),
        )
        self.assertEqual(
            header["environment"]["preflight_kfd_processes"],
            [{"pid": 111, "vram_raw": "0", "vram_bytes": 0}],
        )

    def test_cli_requires_worker_and_fresh_output_directory(self):
        args = TOOL.parse_args(["--worker", "/tmp/worker", "--output-dir", "/tmp/evidence"])
        self.assertEqual(args.worker, Path("/tmp/worker"))
        self.assertEqual(args.output_dir, Path("/tmp/evidence"))
        self.assertEqual(args.amd_smi, "amd-smi")


if __name__ == "__main__":
    unittest.main()
