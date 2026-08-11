from __future__ import annotations

import os
import signal
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import build_rmsnorm_p0_runtime as builder  # noqa: E402
import validate_rmsnorm_p0_contracts as contracts  # noqa: E402
from ci.tests.test_rmsnorm_p0_runner import candidate, prerequisites  # noqa: E402


class P0BuilderTests(unittest.TestCase):
    def test_builder_uses_one_pinned_dedicated_command_and_rechecks_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-builder-") as directory:
            output = Path(directory) / "artifact"

            def build(command: list[str], **kwargs: object) -> object:
                self.assertEqual(command, list(contracts.P0_BUILD_COMMAND))
                environment = kwargs["env"]
                self.assertIsInstance(environment, dict)
                for name, value in contracts.p0_build_environment("gfx1030").items():
                    self.assertEqual(environment[name], value)
                target = Path(environment["CARGO_TARGET_DIR"])
                binary = target / "release" / contracts.P0_BINARY
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"dedicated-p0-binary\n")
                binary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                return builder.subprocess.CompletedProcess(command, 0, b"", b"")

            with patch.object(builder, "_run_bounded_build", side_effect=build) as invoked:
                artifact = builder.build_artifact(
                    repo=ROOT,
                    output_dir=output,
                    candidate=candidate(),
                    target="gfx1030",
                    prerequisites=prerequisites("gfx1030", candidate()),
                )
            self.assertEqual(invoked.call_count, 1)
            self.assertEqual(artifact["binary"]["path"], contracts.P0_BINARY)
            self.assertEqual(artifact["build"]["builder"], "ci/tools/build_rmsnorm_p0_runtime.py")
            self.assertTrue(artifact["build"]["fresh_output"])
            self.assertTrue(artifact["build"]["substitution_rejected"])
            self.assertEqual(artifact["build"]["limits"], contracts.P0_BUILD_LIMITS)
            contracts.validate_artifact(
                artifact,
                ROOT,
                binary_path=output / contracts.P0_BINARY,
            )

    def test_bounded_build_enforces_deadline_output_and_process_group(self) -> None:
        environment = dict(os.environ)
        with patch.object(builder.os, "pidfd_open", side_effect=AssertionError("pidfd used"), create=True):
            success = builder._run_bounded_build(
                [sys.executable, "-c", "import sys; sys.stdout.write('ok')"], cwd=ROOT, env=environment
            )
        self.assertEqual((success.returncode, success.stdout, success.stderr), (0, b"ok", b""))
        nonzero = builder._run_bounded_build(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'out'); sys.stderr.buffer.write(b'err'); sys.exit(7)"],
            cwd=ROOT, env=environment
        )
        self.assertEqual((nonzero.returncode, nonzero.stdout, nonzero.stderr), (7, b"out", b"err"))

        with patch.object(builder, "P0_BUILD_TIMEOUT_SECONDS", 0.05):
            started = time.monotonic()
            with self.assertRaisesRegex(contracts.ContractError, "timed out"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    cwd=ROOT,
                    env=environment,
                )
            self.assertLess(time.monotonic() - started, 5.0)

        with tempfile.TemporaryDirectory(prefix="sllm-p0-build-child-") as directory:
            child_pid = Path(directory) / "child.pid"
            script = (
                "import pathlib, signal, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            )
            with patch.object(builder, "P0_BUILD_TIMEOUT_SECONDS", 0.1):
                with self.assertRaisesRegex(contracts.ContractError, "timed out"):
                    builder._run_bounded_build(
                        [sys.executable, "-c", script, str(child_pid)],
                        cwd=ROOT,
                        env=environment,
                    )
            descendant = int(child_pid.read_text(encoding="ascii"))
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant, 0)

        with patch.object(builder, "P0_BUILD_OUTPUT_LIMIT_BYTES", 1024):
            exact = builder._run_bounded_build(
                [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*512); sys.stderr.buffer.write(b'y'*512)"],
                cwd=ROOT, env=environment
            )
            self.assertEqual(len(exact.stdout) + len(exact.stderr), 1024)
            with self.assertRaisesRegex(contracts.ContractError, "output exceeded"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
                    cwd=ROOT,
                    env=environment,
                )

    def test_eof_with_same_group_child_is_failure_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-eof-child-") as directory:
            child_pid = Path(directory) / "child.pid"
            script = (
                "import os,pathlib,signal,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import os,signal,time; os.close(1); os.close(2); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); sys.exit(0)"
            )
            with patch.object(builder, "P0_BUILD_KILL_GRACE_SECONDS", 0.05):
                with self.assertRaisesRegex(contracts.ContractError, "process group remained"):
                    builder._run_bounded_build(
                        [sys.executable, "-c", script, str(child_pid)],
                        cwd=ROOT,
                        env=dict(os.environ),
                    )
            with self.assertRaises(ProcessLookupError):
                os.kill(int(child_pid.read_text(encoding="ascii")), 0)

    def test_anchor_observation_failure_still_kills_and_reaps(self) -> None:
        with patch.object(builder, "P0_BUILD_KILL_GRACE_SECONDS", 0.05), patch.object(
            builder, "_process_group_members", side_effect=contracts.ContractError("group inspection failed")
        ):
            with self.assertRaisesRegex(contracts.ContractError, "group inspection failed"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "pass"], cwd=ROOT, env=dict(os.environ)
                )

    def test_reap_wait_error_retries_and_sleep_failure_still_kills(self) -> None:
        real_waitpid = builder.os.waitpid
        calls = 0

        def waitpid_once_error(pid: int, options: int) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient reap error")
            return real_waitpid(pid, options)

        with patch.object(builder, "P0_BUILD_TIMEOUT_SECONDS", 0.03), patch.object(
            builder, "P0_BUILD_KILL_GRACE_SECONDS", 0.05
        ), patch.object(builder.os, "waitpid", side_effect=waitpid_once_error):
            with self.assertRaisesRegex(contracts.ContractError, "timed out"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "import time; time.sleep(60)"], cwd=ROOT, env=dict(os.environ)
                )
        self.assertGreaterEqual(calls, 2)

        with patch.object(builder, "P0_BUILD_TIMEOUT_SECONDS", 0.03), patch.object(
            builder, "P0_BUILD_KILL_GRACE_SECONDS", 0.05
        ), patch.object(builder, "_signal_process_group", wraps=builder._signal_process_group) as signals, patch.object(
            builder.time, "sleep", side_effect=OSError("injected sleep failure")
        ):
            with self.assertRaisesRegex(contracts.ContractError, "timed out"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"],
                    cwd=ROOT,
                    env=dict(os.environ),
                )
        self.assertEqual([call.args[1] for call in signals.call_args_list[:2]], [signal.SIGTERM, signal.SIGKILL])

    def test_failed_group_kill_reports_retained_same_session_member(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-kill-failure-") as directory:
            child_pid_path = Path(directory) / "child.pid"
            script = (
                "import pathlib,signal,subprocess,sys,time;child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
            )

            def fail_group_kill(group_id: int, signal_value: signal.Signals, real_signal=builder._signal_process_group) -> None:
                if signal_value == signal.SIGKILL:
                    raise OSError("injected SIGKILL failure")
                real_signal(group_id, signal_value)

            try:
                with patch.object(builder, "P0_BUILD_TIMEOUT_SECONDS", 0.1), patch.object(
                    builder, "P0_BUILD_KILL_GRACE_SECONDS", 0.05
                ), patch.object(builder, "_signal_process_group", side_effect=fail_group_kill):
                    with self.assertRaises(contracts.ContractError) as raised:
                        builder._run_bounded_build(
                            [sys.executable, "-c", script, str(child_pid_path)],
                            cwd=ROOT,
                            env=dict(os.environ),
                        )
                retained_pid = int(child_pid_path.read_text(encoding="ascii"))
                notes = "\n".join(getattr(raised.exception, "__notes__", []))
                self.assertTrue(
                    "timed out" in str(raised.exception)
                    and all(text in notes for text in ("KILL: OSError: injected SIGKILL failure", "post-KILL process group did not disappear; retained members:", str(retained_pid)))
                )
            finally:
                try:
                    os.kill(int(child_pid_path.read_text(encoding="ascii")), signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError, ValueError):
                    pass

    def test_finalizer_closes_each_resource_independently_and_no_pidfd(self) -> None:
        selector, stdout, stderr = Mock(), Mock(), Mock()
        selector.close.side_effect = OSError("close failure")
        errors: list[str] = []
        builder._close_resource(selector, "selector", errors)
        builder._close_resource(stdout, "stdout", errors)
        builder._close_resource(stderr, "stderr", errors)
        self.assertTrue(selector.close.called and stdout.close.called and stderr.close.called)
        self.assertIn("close selector", errors[0])

    def test_builder_refuses_overwrite_and_noncanonical_targets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-builder-") as directory:
            output = Path(directory) / "artifact"
            output.mkdir()
            (output / contracts.P0_BINARY).write_bytes(b"existing")
            with self.assertRaises(contracts.ContractError):
                builder.build_artifact(
                    repo=ROOT,
                    output_dir=output,
                    candidate=candidate(),
                    target="gfx1030",
                    prerequisites=prerequisites("gfx1030", candidate()),
                )
            with self.assertRaises(contracts.ContractError):
                builder.build_artifact(
                    repo=ROOT,
                    output_dir=Path(directory) / "other",
                    candidate=candidate(),
                    target="gfx9999",
                    prerequisites=prerequisites("gfx1030", candidate()),
                )


if __name__ == "__main__":
    unittest.main()
