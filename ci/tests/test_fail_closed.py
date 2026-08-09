#!/usr/bin/env python3
"""Focused direct-runner, network, and fail-closed negative coverage."""

from __future__ import annotations

import json
import argparse
import contextlib
import errno
import io
import os
import signal
import stat
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import run_host_suite as host_runner  # noqa: E402
from common import ContractError  # noqa: E402
from network_guard import (  # noqa: E402
    IsolationPlan,
    LOOPBACK_INIT_SCRIPT,
    NetworkIsolationError,
    SUDO_FALLBACK_TOOL_CANDIDATES,
    _candidate_plans,
    _inspect_system_tool_candidate,
    _normalize_ipv4_routes,
    _normalize_ipv6_routes,
    _select_trusted_system_tool,
    _sudo_fallback_tools,
    _validate_trusted_system_metadata,
    _assert_external_connect_fails,
    _probe,
    assert_isolated,
    child_main,
    prepare_isolation,
    verify_parent_restored,
)
from self_test import run  # noqa: E402


class FailClosedTests(unittest.TestCase):
    def test_invalid_schema_state_zero_collection_and_aggregate_gates_fail(self) -> None:
        # Includes local-development reports being rejected by strict aggregate.
        run()


class RunnerIdentityTests(unittest.TestCase):
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def test_dirty_checkout_requires_explicit_local_opt_in(self) -> None:
        with self.assertRaisesRegex(ContractError, "dirty local"):
            host_runner.validate_execution_identity(
                strict_ci=False,
                allow_dirty_local=False,
                worktree_clean=False,
                head_sha=self.SHA,
                reviewed_sha=self.SHA,
                tested_sha=self.SHA,
                workflow_sha=self.SHA,
            )
        self.assertEqual(
            host_runner.validate_execution_identity(
                strict_ci=False,
                allow_dirty_local=True,
                worktree_clean=False,
                head_sha=self.SHA,
                reviewed_sha=self.SHA,
                tested_sha=self.SHA,
                workflow_sha=self.SHA,
            ),
            "local-development",
        )

    def test_strict_identity_requires_exact_matching_sha(self) -> None:
        with self.assertRaisesRegex(ContractError, "exactly match"):
            host_runner.validate_execution_identity(
                strict_ci=True,
                allow_dirty_local=False,
                worktree_clean=True,
                head_sha=self.SHA,
                reviewed_sha=self.SHA[:-1] + "0",
                tested_sha=self.SHA,
                workflow_sha=self.SHA,
            )

    def test_unregistered_and_traversal_commands_are_rejected(self) -> None:
        valid = [sys.executable, "ci/tests/test_fail_closed.py"]
        wrapped = host_runner.execution_argv(valid, repo=ROOT)
        self.assertEqual(wrapped[0], sys.executable)
        self.assertEqual(wrapped[2], host_runner.UNITTEST_WRAPPER_FLAG)
        invalid = [sys.executable, "../ci/tests/test_fail_closed.py"]
        with self.assertRaises(ValueError):
            host_runner.execution_argv(invalid, repo=ROOT)
        with self.assertRaises(ValueError):
            host_runner.execution_argv(
                [sys.executable, "ci/tests/test_fail_closed.py", "RunnerIdentityTests"],
                repo=ROOT,
            )
        with self.assertRaises(ValueError):
            host_runner.execution_argv(
                [sys.executable, "-m", "unittest", "discover", "ci/tests"],
                repo=ROOT,
            )

    def test_unittest_registry_requires_exact_argv_identity(self) -> None:
        unregistered_shapes = [
            [sys.executable, "-m", "unittest", "ci.tests.test_fail_closed"],
            [sys.executable, "-m", "unittest"],
            [sys.executable, "-m", "unittest", "ci.tests.test_fail_closed", "RunnerIdentityTests"],
            [sys.executable, "-m", "unittest", "ci/tests/test_fail_closed.py"],
            [sys.executable, "ci/tests/test_fail_closed.py", "--"],
            [sys.executable, "ci/tests/./test_fail_closed.py"],
            [sys.executable, "ci/tests/test_fail_closed.py", "RunnerIdentityTests.test_dirty_checkout_requires_explicit_local_opt_in"],
            [sys.executable, "ci/tests/not_registered.py"],
            [sys.executable, "/tmp/ci/tests/test_fail_closed.py"],
            [sys.executable, "ci/tests/../tests/test_fail_closed.py"],
        ]
        for argv in unregistered_shapes:
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError):
                    host_runner.execution_argv(argv, repo=ROOT)
                self.assertFalse(host_runner._is_registered_unittest_invocation(argv, ROOT))

    def test_every_registered_host_command_has_exact_direct_or_wrapper_classification(self) -> None:
        commands = host_runner._registered_host_commands(ROOT)
        unittest_commands = host_runner._registered_unittest_commands(ROOT)
        self.assertEqual(len(commands), 29)
        self.assertEqual(len(unittest_commands), 13)
        self.assertEqual(len(commands) - len(unittest_commands), 16)
        validator_commands = [
            command for command in commands
            if len(command) > 1 and command[1].startswith("ci/tools/")
        ]
        self.assertEqual(len(validator_commands), 13)
        self.assertTrue(all(host_runner.execution_argv(command, repo=ROOT) == command for command in validator_commands))
        for command in commands:
            with self.subTest(command=command):
                if command in unittest_commands:
                    wrapped = host_runner.execution_argv(command, repo=ROOT)
                    self.assertEqual(wrapped[:3], [
                        sys.executable,
                        str((ROOT / "ci/tools/run_host_suite.py").resolve()),
                        host_runner.UNITTEST_WRAPPER_FLAG,
                    ])
                    self.assertTrue(host_runner.is_unittest_script(command, repo=ROOT))
                else:
                    self.assertEqual(host_runner.execution_argv(command, repo=ROOT), command)
                    self.assertFalse(host_runner.is_unittest_script(command, repo=ROOT))
                if len(command) > 1 and command[1].startswith("ci/tools/"):
                    counts, warning, source = host_runner.actual_counts(command, "", 0, repo=ROOT)
                    self.assertEqual(source, "validator-command")
                    self.assertIsNone(warning)
                    self.assertEqual(counts["selected"], 1)

    def test_systematic_unregistered_direct_module_and_script_variants_are_rejected(self) -> None:
        commands = host_runner._registered_host_commands(ROOT)
        variants: list[list[str]] = []
        for command in commands:
            variants.append([*command, "--unregistered"])
            if len(command) >= 3 and command[1] == "-m":
                module_variant = command.copy()
                module_variant[2] = f"{module_variant[2]}.unregistered"
                variants.append(module_variant)
            elif len(command) >= 2 and command[1].endswith(".py"):
                script_variant = command.copy()
                script_variant[1] = f"{script_variant[1]}.unregistered"
                variants.append(script_variant)
        registered = {tuple(command) for command in commands}
        for variant in variants:
            with self.subTest(argv=variant):
                self.assertNotIn(tuple(variant), registered)
                with self.assertRaises(ValueError):
                    host_runner.execution_argv(variant, repo=ROOT)

    def test_incomplete_row_at_exact_output_limit_is_a_breach(self) -> None:
        limit = 257
        self.assertFalse(host_runner.row_output_breach(
            aggregate_output=limit - 1,
            max_row_output_bytes=limit,
            commands_complete=False,
        ))
        self.assertTrue(host_runner.row_output_breach(
            aggregate_output=limit,
            max_row_output_bytes=limit,
            commands_complete=False,
        ))
        self.assertTrue(host_runner.row_output_breach(
            aggregate_output=limit + 1,
            max_row_output_bytes=limit,
            commands_complete=True,
        ))
        self.assertFalse(host_runner.row_output_breach(
            aggregate_output=limit,
            max_row_output_bytes=limit,
            commands_complete=True,
        ))

    def test_actual_counts_require_one_consistent_record(self) -> None:
        output = (
            'SLLM_UNITTEST_COUNTS=' +
            json.dumps({
                "collected": 3, "selected": 3, "passed": 2,
                "failed": 0, "skipped": 1, "deselected": 0,
            }, separators=(",", ":"))
        )
        counts, warning, source = host_runner.actual_counts(
            [sys.executable, "ci/tests/test_fail_closed.py"], output, 0, repo=ROOT
        )
        self.assertEqual(source, "unittest-machine")
        self.assertIsNone(warning)
        self.assertEqual(counts["selected"], 3)
        _, warning, _ = host_runner.actual_counts(
            [sys.executable, "ci/tests/test_fail_closed.py"], output + "\n" + output, 0, repo=ROOT
        )
        self.assertIn("exactly one", warning or "")

    def test_registered_validator_command_uses_single_validator_count(self) -> None:
        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        counts, warning, source = host_runner.actual_counts(command, "", 0, repo=ROOT)
        self.assertEqual(source, "validator-command")
        self.assertIsNone(warning)
        self.assertEqual(counts["selected"], 1)


class DirectBoundedProcessTests(unittest.TestCase):
    def _run(self, code: str, *, timeout: float = 1.0, output_limit: int = 4096, rss: int = 512 * 1024 * 1024):
        return host_runner.run_bounded_process(
            [sys.executable, "-B", "-c", code],
            repo=ROOT,
            timeout_seconds=timeout,
            max_rss_bytes=rss,
            output_limit_bytes=output_limit,
            deadline=time.monotonic() + timeout,
        )

    def test_timeout_is_bounded_and_process_group_is_reaped(self) -> None:
        result = self._run("import time; time.sleep(2)", timeout=0.08)
        self.assertTrue(result[4])
        self.assertIsNotNone(result[2])

    @staticmethod
    def _process_state(pid: int) -> str | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except OSError:
            return None
        command_end = stat.rfind(")")
        fields = stat[command_end + 2 :].split() if command_end >= 0 else []
        return fields[0] if fields else None

    def test_successful_leader_exit_does_not_leave_live_fork_descendant(self) -> None:
        result = self._run(
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.close(1); os.close(2); time.sleep(30); os._exit(0)\n"
            "print(pid, flush=True)",
        )
        descendant_pid = int(result[0].decode("ascii").strip())
        state = self._process_state(descendant_pid)
        try:
            self.assertEqual(result[2], 0)
            self.assertFalse(result[4])
            self.assertTrue(
                state is None or state in {"Z", "X", "x"},
                f"runner returned with live descendant {descendant_pid} in state {state}",
            )
        finally:
            if state is not None and state not in {"Z", "X", "x"}:
                os.kill(descendant_pid, signal.SIGKILL)

    def test_procfs_snapshot_ignores_only_transient_pid_disappearance(self) -> None:
        entry = Path("/proc/123")
        fields = [b"S", b"1", b"777", *([b"0"] * 18), b"1"]
        valid_stat = b"123 (snapshot test) " + b" ".join(fields)
        with (
            patch.object(Path, "iterdir", return_value=[entry]),
            patch.object(Path, "read_bytes", return_value=valid_stat),
        ):
            snapshot = host_runner.process_group_snapshot(777)
        self.assertEqual(snapshot.live_members, (123,))
        self.assertEqual(snapshot.live_rss_bytes, os.sysconf("SC_PAGE_SIZE"))

        for transient_errno in (errno.ENOENT, errno.ESRCH):
            with self.subTest(transient_errno=transient_errno):
                with (
                    patch.object(Path, "iterdir", return_value=[entry]),
                    patch.object(
                        Path,
                        "read_bytes",
                        side_effect=OSError(transient_errno, "process disappeared"),
                    ),
                ):
                    self.assertEqual(host_runner.process_group_snapshot(777).members, ())

        failures = (
            PermissionError(errno.EACCES, "permission denied"),
            OSError(errno.EIO, "I/O failure"),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                with (
                    patch.object(Path, "iterdir", return_value=[entry]),
                    patch.object(Path, "read_bytes", side_effect=failure),
                ):
                    with self.assertRaises(OSError):
                        host_runner.process_group_snapshot(777)

        with patch.object(Path, "iterdir", side_effect=PermissionError(errno.EACCES, "hidden procfs")):
            with self.assertRaises(PermissionError):
                host_runner.process_group_snapshot(777)
        with (
            patch.object(Path, "iterdir", return_value=[entry]),
            patch.object(Path, "read_bytes", return_value=b"123 malformed"),
        ):
            with self.assertRaisesRegex(OSError, "malformed procfs stat"):
                host_runner.process_group_snapshot(777)

    def test_procfs_inspection_failure_still_reaps_process_group_leader(self) -> None:
        real_popen = subprocess.Popen
        children: list[subprocess.Popen[bytes]] = []

        def capture_popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        with (
            patch.object(host_runner.subprocess, "Popen", side_effect=capture_popen),
            patch.object(
                host_runner,
                "process_group_snapshot",
                side_effect=PermissionError(errno.EACCES, "procfs inspection denied"),
            ),
        ):
            with self.assertRaises(PermissionError):
                self._run("import time; time.sleep(30)", timeout=0.5)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        state = self._process_state(children[0].pid)
        self.assertTrue(state is None or state in {"Z", "X", "x"})

    def test_selector_setup_failure_closes_and_reaps_child(self) -> None:
        real_popen = subprocess.Popen

        class RegisterFailSelector:
            def __init__(self) -> None:
                self.register_calls = 0
                self.closed = False

            def register(self, *_args) -> None:
                self.register_calls += 1
                if self.register_calls == 2:
                    raise OSError(errno.EIO, "injected selector registration failure")

            def close(self) -> None:
                self.closed = True

        for stage in ("construction", "registration"):
            with self.subTest(stage=stage):
                children: list[subprocess.Popen[bytes]] = []
                injected_selector = RegisterFailSelector()

                def capture_popen(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    return child

                selector_patch = (
                    patch.object(
                        host_runner.selectors,
                        "DefaultSelector",
                        side_effect=OSError(errno.EIO, "injected selector construction failure"),
                    )
                    if stage == "construction"
                    else patch.object(
                        host_runner.selectors,
                        "DefaultSelector",
                        return_value=injected_selector,
                    )
                )
                with (
                    patch.object(host_runner.subprocess, "Popen", side_effect=capture_popen),
                    selector_patch,
                ):
                    with self.assertRaises(OSError):
                        self._run("import time; time.sleep(30)", timeout=0.5)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].returncode)
                self.assertIsNotNone(children[0].stdout)
                self.assertIsNotNone(children[0].stderr)
                self.assertTrue(children[0].stdout.closed)
                self.assertTrue(children[0].stderr.closed)
                state = self._process_state(children[0].pid)
                self.assertTrue(state is None or state in {"Z", "X", "x"})
                if stage == "registration":
                    self.assertTrue(injected_selector.closed)

    def test_pipe_eio_does_not_accept_zero_exit_with_zero_observed_output(self) -> None:
        real_popen = subprocess.Popen
        real_read = os.read
        children: list[subprocess.Popen[bytes]] = []
        pipe_fds: set[int] = set()
        injected = False

        def capture_popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            assert child.stdout is not None and child.stderr is not None
            pipe_fds.update((child.stdout.fileno(), child.stderr.fileno()))
            return child

        def inject_eio(fd: int, size: int) -> bytes:
            nonlocal injected
            if fd in pipe_fds and not injected:
                injected = True
                raise OSError(errno.EIO, "injected pipe read failure")
            return real_read(fd, size)

        with (
            patch.object(host_runner.subprocess, "Popen", side_effect=capture_popen),
            patch.object(host_runner.os, "read", side_effect=inject_eio),
        ):
            with self.assertRaisesRegex(OSError, "injected pipe read failure"):
                self._run("pass", timeout=0.5)

        self.assertTrue(injected)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].returncode, 0)
        self.assertTrue(children[0].stdout is not None and children[0].stdout.closed)
        self.assertTrue(children[0].stderr is not None and children[0].stderr.closed)

    def test_rss_limit_includes_reparented_double_fork_group_member(self) -> None:
        result = self._run(
            "import os,time\n"
            "read_fd,write_fd=os.pipe()\n"
            "first=os.fork()\n"
            "if first == 0:\n"
            " second=os.fork()\n"
            " if second == 0:\n"
            "  os.close(read_fd)\n"
            "  payload=bytearray(96 * 1024 * 1024)\n"
            "  for offset in range(0, len(payload), 4096): payload[offset]=1\n"
            "  os.write(write_fd, b'1')\n"
            "  os.close(write_fd); os.close(1); os.close(2); time.sleep(30)\n"
            "  os._exit(0)\n"
            " os._exit(0)\n"
            "os.close(write_fd); os.waitpid(first, 0); os.read(read_fd, 1)\n"
            "os.close(read_fd); time.sleep(0.4)\n",
            timeout=2.0,
            rss=40 * 1024 * 1024,
        )
        self.assertTrue(result[6])
        self.assertGreater(result[7], 40 * 1024 * 1024)

    def test_output_limit_is_fail_closed(self) -> None:
        result = self._run("import sys; sys.stdout.write('x' * 8192)", output_limit=128)
        self.assertTrue(result[5])
        self.assertGreater(result[8], 128)
        self.assertLessEqual(len(result[0]), 128)

    def test_rss_limit_is_fail_closed(self) -> None:
        result = self._run(
            "x = bytearray(8 * 1024 * 1024); import time; time.sleep(0.2)",
            rss=1024 * 1024,
        )
        self.assertTrue(result[6])


class NetworkSetupDeadlineTests(unittest.TestCase):
    def _run_with_restoration_mismatch(
        self,
        bounded_result: tuple[bytes, bytes, int, float, bool, bool, bool, int, int, int, float, float],
        *,
        max_rss_bytes: int = 512 * 1024 * 1024,
        output_limit_bytes: int = 4096,
    ) -> dict[str, object]:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        mismatch = NetworkIsolationError("test execution changed the parent network namespace")
        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        with (
            patch.object(host_runner, "prepare_isolation", return_value=plan),
            patch.object(host_runner, "wrap_command", return_value=command),
            patch.object(host_runner, "run_bounded_process", return_value=bounded_result),
            patch.object(host_runner, "verify_parent_restored", side_effect=mismatch),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes"),
        ):
            step, _ = host_runner.run_command(
                "combined-restoration-mismatch",
                command,
                timeout_seconds=1.0,
                repo=ROOT,
                output_dir=ROOT / ".not-written",
                max_rss_bytes=max_rss_bytes,
                output_limit_bytes=output_limit_bytes,
                address_space_limit_bytes=None,
            )
        return step

    def test_parent_restoration_mismatch_precedes_execution_timeout(self) -> None:
        step = self._run_with_restoration_mismatch(
            (b"", b"", 0, 0.01, True, False, False, 1024, 0, 0, 0.0, 0.0)
        )
        self.assertEqual(step["state"], "INFRA_ERROR")
        self.assertTrue(step["resource"]["timed_out"])
        self.assertIn("parent network namespace", step["diagnostic"])
        self.assertIn("command timeout", step["diagnostic"])

    def test_execution_timeout_with_restored_parent_remains_fail(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        bounded_result = (
            b"", b"", 0, 1.0, True, False, False, 1024, 0, 0, 0.0, 0.0,
        )
        with (
            patch.object(host_runner, "prepare_isolation", return_value=plan),
            patch.object(host_runner, "wrap_command", return_value=command),
            patch.object(host_runner, "run_bounded_process", return_value=bounded_result),
            patch.object(host_runner, "verify_parent_restored") as verify,
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes"),
        ):
            step, _ = host_runner.run_command(
                "execution-timeout-restored-parent",
                command,
                timeout_seconds=1.0,
                repo=ROOT,
                output_dir=ROOT / ".not-written",
                max_rss_bytes=512 * 1024 * 1024,
                output_limit_bytes=4096,
                address_space_limit_bytes=None,
            )
        self.assertEqual(step["state"], "FAIL")
        self.assertTrue(step["resource"]["timed_out"])
        self.assertIn("command timeout", step["diagnostic"])
        self.assertNotIn("network isolation failed", step["diagnostic"])
        verify.assert_called_once_with(plan, outer_deadline=None)

    def test_parent_restoration_mismatch_precedes_output_and_rss_breaches(self) -> None:
        rss_limit = 40 * 1024 * 1024
        observed_rss = 96 * 1024 * 1024
        step = self._run_with_restoration_mismatch(
            (b"x", b"", 0, 0.01, False, True, True, observed_rss, 8192, 0, 0.0, 0.0),
            max_rss_bytes=rss_limit,
            output_limit_bytes=4096,
        )
        self.assertEqual(step["state"], "INFRA_ERROR")
        self.assertTrue(step["resource"]["output_breach"])
        self.assertTrue(step["resource"]["rss_breach"])
        self.assertIn("parent network namespace", step["diagnostic"])
        self.assertIn("command output exceeded", step["diagnostic"])
        self.assertIn("max RSS", step["diagnostic"])

    def test_delayed_network_setup_is_bounded_and_fail_closed(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        observed: dict[str, float] = {}

        def delayed_setup(*, outer_deadline: float) -> IsolationPlan:
            observed["deadline"] = outer_deadline
            time.sleep(0.02)
            return plan

        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        started = time.monotonic()
        with (
            patch.object(host_runner, "prepare_isolation", side_effect=delayed_setup),
            patch.object(host_runner, "verify_parent_restored") as verify,
            patch.object(host_runner, "run_bounded_process") as bounded,
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes"),
        ):
            step, _ = host_runner.run_command(
                "mocked-setup-deadline",
                command,
                timeout_seconds=0.005,
                repo=ROOT,
                output_dir=ROOT / ".not-written",
                max_rss_bytes=512 * 1024 * 1024,
                output_limit_bytes=4096,
                address_space_limit_bytes=None,
            )
        elapsed = time.monotonic() - started
        self.assertIn("deadline", observed)
        self.assertLess(observed["deadline"] - started, 0.1)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(step["state"], "FAIL")
        self.assertTrue(step["resource"]["timed_out"])
        self.assertIn("setup exhausted command wall time", step["diagnostic"])
        bounded.assert_not_called()
        verify.assert_called_once()

    def test_launch_deadline_with_restored_parent_is_timeout_without_popen(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        clock = [0.0]

        def delayed_env() -> dict[str, str]:
            clock[0] = 2.0
            return {}

        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        with (
            patch.object(host_runner, "prepare_isolation", return_value=plan),
            patch.object(host_runner, "wrap_command", return_value=command),
            patch.object(host_runner, "isolated_env", side_effect=delayed_env),
            patch.object(host_runner, "verify_parent_restored") as verify,
            patch.object(host_runner.subprocess, "Popen") as popen,
            patch.object(host_runner.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes"),
        ):
            step, _ = host_runner.run_command(
                "bounded-env-deadline",
                command,
                timeout_seconds=1.0,
                repo=ROOT,
                output_dir=ROOT / ".not-written",
                max_rss_bytes=512 * 1024 * 1024,
                output_limit_bytes=4096,
                address_space_limit_bytes=None,
            )
        self.assertEqual(step["state"], "FAIL")
        self.assertTrue(step["resource"]["timed_out"])
        self.assertIn("process launch deadline expired", step["diagnostic"])
        self.assertNotIn("INFRA_ERROR", step["diagnostic"])
        verify.assert_called_once_with(plan, outer_deadline=None)
        popen.assert_not_called()

    def test_launch_deadline_with_parent_mismatch_is_infra_without_popen(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        clock = [0.0]

        def delayed_env() -> dict[str, str]:
            clock[0] = 2.0
            return {}

        mismatch = NetworkIsolationError("test execution changed the parent network namespace")
        command = [sys.executable, "ci/tools/validate_python.py", "--mode", "compile"]
        with (
            patch.object(host_runner, "prepare_isolation", return_value=plan),
            patch.object(host_runner, "wrap_command", return_value=command),
            patch.object(host_runner, "isolated_env", side_effect=delayed_env),
            patch.object(host_runner, "verify_parent_restored", side_effect=mismatch) as verify,
            patch.object(host_runner.subprocess, "Popen") as popen,
            patch.object(host_runner.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes"),
        ):
            step, _ = host_runner.run_command(
                "bounded-env-restoration-mismatch",
                command,
                timeout_seconds=1.0,
                repo=ROOT,
                output_dir=ROOT / ".not-written",
                max_rss_bytes=512 * 1024 * 1024,
                output_limit_bytes=4096,
                address_space_limit_bytes=None,
            )
        self.assertEqual(step["state"], "INFRA_ERROR")
        self.assertFalse(step["resource"]["timed_out"])
        self.assertIn("parent network namespace", step["diagnostic"])
        self.assertNotIn("process launch deadline expired", step["diagnostic"])
        verify.assert_called_once_with(plan, outer_deadline=None)
        popen.assert_not_called()

    def test_delayed_probe_cannot_return_success_after_deadline(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        with (
            patch("network_guard.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")) as run,
            patch("network_guard.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
        ):
            passed, detail = _probe(plan, outer_deadline=1.0)
        self.assertFalse(passed)
        self.assertIn("deadline expired", detail)
        self.assertEqual(run.call_args.kwargs["timeout"], 1.0)

    def test_prepare_rechecks_deadline_after_delayed_probe(self) -> None:
        plan = IsolationPlan(
            strategy="mocked",
            prefix=(),
            parent_netns="net:[4026531840]",
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )
        clock = [0.0]

        def delayed_probe(_plan: IsolationPlan, *, outer_deadline: float) -> tuple[bool, str]:
            self.assertEqual(outer_deadline, 1.0)
            clock[0] = 2.0
            return True, ""

        with (
            patch.dict(os.environ, {"SLLM_NETWORK_GUARD_ACTIVE": "0"}),
            patch("network_guard.time.monotonic", side_effect=lambda: clock[0]),
            patch("network_guard.current_netns", side_effect=[plan.parent_netns, plan.parent_netns]),
            patch("network_guard._candidate_plans", return_value=[plan]),
            patch("network_guard._probe", side_effect=delayed_probe),
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "deadline expired"):
                prepare_isolation(outer_deadline=1.0)


class NetworkRouteNormalizationTests(unittest.TestCase):
    IPV4_HEADER = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT"
    IPV4_FIELDS = ["enp7s0", "0A0B0C0D", "01020304", "0003", "255", "256", "257", "00FFFFFF", "1501", "3", "65535"]
    IPV6_FIELDS = [
        "20010DB8000000000000000000000001", "41",
        "20010DB8000000000000000000000002", "07",
        "00000000000000000000000000000003", "00000101",
        "000000FF", "00000100", "00000005", "enp7s0",
    ]

    def test_ipv4_empty_input_fails_closed_but_header_only_is_empty_table(self) -> None:
        with self.assertRaisesRegex(NetworkIsolationError, "missing header"):
            _normalize_ipv4_routes([])
        self.assertEqual(_normalize_ipv4_routes([self.IPV4_HEADER]), ())

    def test_sudo_fallback_initializes_loopback_before_privilege_drop(self) -> None:
        tools = _sudo_fallback_tools()
        self.assertIsNotNone(tools)
        assert tools is not None
        repository_unshare = str(ROOT / "repository-controlled-unshare")
        with patch("network_guard.shutil.which", return_value=repository_unshare) as which:
            plans = _candidate_plans("net:[4026531840]")
        which.assert_called_once_with("unshare")
        fallback = next(
            plan for plan in plans
            if plan.strategy == "sudo-network-namespace-drop-privileges"
        )
        self.assertTrue(fallback.require_no_capabilities)
        self.assertEqual(
            fallback.prefix[:10],
            (
                tools["sudo"], "-n", tools["unshare"], "--net", "--fork",
                tools["shell"], "-c", LOOPBACK_INIT_SCRIPT, "sllm-loopback-init",
                tools["ip"],
            ),
        )
        self.assertNotIn(repository_unshare, fallback.prefix)
        setpriv_index = fallback.prefix.index(tools["setpriv"])
        self.assertEqual(setpriv_index, 10)
        self.assertIn("--clear-groups", fallback.prefix[setpriv_index + 1 :])
        self.assertIn("--bounding-set=-all", fallback.prefix[setpriv_index + 1 :])
        self.assertIn("--no-new-privs", fallback.prefix[setpriv_index + 1 :])

    def test_counter_changes_are_ignored_but_semantic_changes_are_not(self) -> None:
        baseline = _normalize_ipv4_routes([self.IPV4_HEADER, " ".join(self.IPV4_FIELDS)])
        counters_changed = self.IPV4_FIELDS.copy()
        counters_changed[4] = "256"
        counters_changed[5] = "257"
        self.assertEqual(
            baseline,
            _normalize_ipv4_routes([self.IPV4_HEADER, " ".join(counters_changed)]),
        )

        semantic_changes = {
            0: "enp8s0",
            1: "0A0B0C0E",
            2: "01020305",
            3: "0007",
            6: "256",
            7: "00FFFF00",
            8: "1500",
            9: "4",
            10: "65534",
        }
        for index, value in semantic_changes.items():
            with self.subTest(protocol="IPv4", field=index):
                changed = self.IPV4_FIELDS.copy()
                changed[index] = value
                self.assertNotEqual(
                    baseline,
                    _normalize_ipv4_routes([self.IPV4_HEADER, " ".join(changed)]),
                )

        baseline6 = _normalize_ipv6_routes([" ".join(self.IPV6_FIELDS)])
        counters_changed6 = self.IPV6_FIELDS.copy()
        counters_changed6[6] = "00000100"
        counters_changed6[7] = "00000101"
        self.assertEqual(baseline6, _normalize_ipv6_routes([" ".join(counters_changed6)]))

        semantic_changes6 = {
            0: "20010DB8000000000000000000000003",
            1: "42",
            2: "20010DB8000000000000000000000004",
            3: "08",
            4: "00000000000000000000000000000004",
            5: "00000100",
            8: "00000006",
            9: "enp8s0",
        }
        for index, value in semantic_changes6.items():
            with self.subTest(protocol="IPv6", field=index):
                changed = self.IPV6_FIELDS.copy()
                changed[index] = value
                self.assertNotEqual(
                    baseline6,
                    _normalize_ipv6_routes([" ".join(changed)]),
                )

    def test_malformed_routes_fail_closed(self) -> None:
        ipv4 = self.IPV4_FIELDS.copy()
        ipv6 = self.IPV6_FIELDS.copy()
        invalid_ipv4_cases = [
            [self.IPV4_HEADER.replace("Use", "Uses"), " ".join(ipv4)],
            [self.IPV4_HEADER, " ".join(ipv4[:-1])],
        ]
        invalid_ipv4_hex = ipv4.copy()
        invalid_ipv4_hex[1] = "not-hex!"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_hex)])
        invalid_ipv4_decimal = ipv4.copy()
        invalid_ipv4_decimal[6] = "0x101"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_decimal)])
        invalid_ipv4_range = ipv4.copy()
        invalid_ipv4_range[8] = "4294967296"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_range)])
        invalid_ipv4_counter = ipv4.copy()
        invalid_ipv4_counter[4] = "0x100"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_counter)])
        for lines in invalid_ipv4_cases:
            with self.subTest(protocol="IPv4", lines=lines):
                with self.assertRaises(NetworkIsolationError):
                    _normalize_ipv4_routes(lines)

        invalid_ipv6_cases = [ipv6[:-1], ipv6 + ["extra"]]
        invalid_ipv6_hex = ipv6.copy()
        invalid_ipv6_hex[4] = "not-hex"
        invalid_ipv6_cases.append(invalid_ipv6_hex)
        invalid_ipv6_prefix = ipv6.copy()
        invalid_ipv6_prefix[1] = "FF"
        invalid_ipv6_cases.append(invalid_ipv6_prefix)
        invalid_ipv6_counter = ipv6.copy()
        invalid_ipv6_counter[6] = "0x100"
        invalid_ipv6_cases.append(invalid_ipv6_counter)
        for fields in invalid_ipv6_cases:
            with self.subTest(protocol="IPv6", fields=fields):
                with self.assertRaises(NetworkIsolationError):
                    _normalize_ipv6_routes([" ".join(fields)])


class SudoFallbackSystemToolTests(unittest.TestCase):
    @staticmethod
    def _metadata(mode: int, *, uid: int = 0, device: int = 7, inode: int = 11) -> os.stat_result:
        return os.stat_result((mode, inode, device, 1, uid, 0, 0, 0, 0, 0))

    def test_non_root_owned_writable_and_nonregular_tools_fail_closed(self) -> None:
        invalid = (
            (self._metadata(stat.S_IFREG | 0o755, uid=1000), "not root-owned"),
            (self._metadata(stat.S_IFREG | 0o775), "group/world writable"),
            (self._metadata(stat.S_IFREG | 0o757), "group/world writable"),
            (self._metadata(stat.S_IFDIR | 0o755), "not a regular file"),
        )
        for role, _ in SUDO_FALLBACK_TOOL_CANDIDATES:
            for metadata, message in invalid:
                with self.subTest(role=role, message=message):
                    with self.assertRaisesRegex(NetworkIsolationError, message):
                        _validate_trusted_system_metadata(
                            role,
                            Path(f"/usr/bin/{role}"),
                            metadata,
                            require_regular=True,
                        )

    def test_missing_fixed_tool_fails_closed(self) -> None:
        with patch("network_guard.os.lstat", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(NetworkIsolationError, "unavailable at fixed system paths"):
                _select_trusted_system_tool("sudo", ("/usr/bin/sudo",))

    def test_each_fixed_tool_is_mandatory_for_sudo_fallback(self) -> None:
        for missing_role, _ in SUDO_FALLBACK_TOOL_CANDIDATES:
            def select(role: str, _candidates: tuple[str, ...]) -> str:
                if role == missing_role:
                    raise NetworkIsolationError(f"missing {role}")
                return f"/usr/bin/{role}"

            with self.subTest(missing_role=missing_role):
                with patch("network_guard._select_trusted_system_tool", side_effect=select):
                    self.assertIsNone(_sudo_fallback_tools())

    def test_unresolved_symlink_and_non_root_symlink_fail_closed(self) -> None:
        root_symlink = self._metadata(stat.S_IFLNK | 0o777)
        with (
            patch("network_guard.os.lstat", return_value=root_symlink),
            patch.object(Path, "resolve", side_effect=RuntimeError("symlink loop")),
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "unresolved or ambiguous symlink"):
                _inspect_system_tool_candidate("shell", "/bin/sh")

        non_root_symlink = self._metadata(stat.S_IFLNK | 0o777, uid=1000)
        with patch("network_guard.os.lstat", return_value=non_root_symlink):
            with self.assertRaisesRegex(NetworkIsolationError, "not root-owned"):
                _inspect_system_tool_candidate("shell", "/bin/sh")

    def test_same_inode_system_aliases_are_accepted(self) -> None:
        identity = (7, 11)
        with patch(
            "network_guard._inspect_system_tool_candidate",
            side_effect=[("/usr/bin/dash", identity), ("/usr/bin/dash", identity)],
        ):
            selected = _select_trusted_system_tool("shell", ("/bin/sh", "/usr/bin/sh"))
        self.assertEqual(selected, "/usr/bin/dash")

    def test_distinct_alias_identities_are_rejected_as_ambiguous(self) -> None:
        with patch(
            "network_guard._inspect_system_tool_candidate",
            side_effect=[("/usr/bin/dash", (7, 11)), ("/usr/bin/bash", (7, 12))],
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "ambiguous tool identities"):
                _select_trusted_system_tool("shell", ("/bin/sh", "/usr/bin/sh"))

    def test_invalid_sudo_tools_do_not_break_user_namespace_plan(self) -> None:
        ambient_unshare = str(ROOT / "repository-controlled-unshare")
        with (
            patch("network_guard.shutil.which", return_value=ambient_unshare),
            patch("network_guard._sudo_fallback_tools", return_value=None),
        ):
            plans = _candidate_plans("net:[4026531840]")
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy, "user-network-namespace")
        self.assertEqual(plans[0].prefix[0], ambient_unshare)


class NetworkNamespaceRestorationTests(unittest.TestCase):
    def _plan(self, parent_netns: str = "net:[4026531840]") -> IsolationPlan:
        return IsolationPlan(
            strategy="test",
            prefix=(),
            parent_netns=parent_netns,
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )

    def test_parent_netns_restoration_is_checked(self) -> None:
        plan = self._plan()
        with patch("network_guard.current_netns", return_value=plan.parent_netns):
            verify_parent_restored(plan)
        with patch("network_guard.current_netns", return_value="net:[4026531841]"):
            with self.assertRaisesRegex(NetworkIsolationError, "parent network namespace"):
                verify_parent_restored(plan)

    def test_parent_netns_mismatch_precedes_expired_deadline(self) -> None:
        plan = self._plan()
        with patch("network_guard.current_netns", return_value="net:[4026531841]"):
            with self.assertRaisesRegex(NetworkIsolationError, "parent network namespace"):
                verify_parent_restored(plan, outer_deadline=0.0)

    def test_restored_parent_with_expired_deadline_reports_deadline(self) -> None:
        plan = self._plan()
        with patch("network_guard.current_netns", return_value=plan.parent_netns):
            with self.assertRaisesRegex(NetworkIsolationError, "network isolation deadline expired"):
                verify_parent_restored(plan, outer_deadline=0.0)

    def test_parent_change_during_probe_fails_closed(self) -> None:
        plan = self._plan()
        with (
            patch.dict(os.environ, {"SLLM_NETWORK_GUARD_ACTIVE": "0"}),
            patch("network_guard.current_netns", side_effect=[plan.parent_netns, "net:[4026531841]"]),
            patch("network_guard._candidate_plans", return_value=[plan]),
            patch("network_guard._probe", return_value=(True, "")),
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "parent network namespace"):
                prepare_isolation()

    def test_delayed_child_inspection_cannot_reach_exec_after_deadline(self) -> None:
        clock = [0.0]

        def delayed_routes() -> tuple[tuple[()], tuple[()]]:
            clock[0] = 2.0
            return ((), ())

        args = argparse.Namespace(
            execution_env=[],
            address_space_limit_bytes=None,
            parent_netns="net:[4026531840]",
            expected_euid=1000,
            expected_egid=1000,
            require_no_capabilities=False,
            outer_deadline=1.0,
            probe=False,
            command=["--", sys.executable, "-c", "pass"],
            strategy="mocked",
        )
        with (
            patch("network_guard.time.monotonic", side_effect=lambda: clock[0]),
            patch("network_guard.resource.setrlimit"),
            patch("network_guard.current_netns", return_value="net:[4026531841]"),
            patch("network_guard.os.geteuid", return_value=1000),
            patch("network_guard.os.getegid", return_value=1000),
            patch("network_guard._interface_names", return_value=("lo",)),
            patch("network_guard._route_snapshot", side_effect=delayed_routes),
            patch("network_guard._assert_external_connect_fails"),
            patch("network_guard.os.execvpe") as execvpe,
        ):
            result = child_main(args)
        self.assertEqual(result, 2)
        execvpe.assert_not_called()

    def test_delayed_child_env_creation_has_clean_timeout_failure(self) -> None:
        clock = [0.0]

        class DelayedEnvironment(dict[str, str]):
            def copy(self) -> dict[str, str]:
                clock[0] = 2.0
                return {}

        args = argparse.Namespace(
            execution_env=[],
            address_space_limit_bytes=None,
            parent_netns="net:[4026531840]",
            expected_euid=1000,
            expected_egid=1000,
            require_no_capabilities=False,
            outer_deadline=1.0,
            probe=False,
            command=["--", sys.executable, "-c", "pass"],
            strategy="mocked",
        )
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            patch("network_guard.time.monotonic", side_effect=lambda: clock[0]),
            patch("network_guard.os.environ", DelayedEnvironment()),
            patch("network_guard.resource.setrlimit"),
            patch("network_guard.current_netns", return_value="net:[4026531841]"),
            patch("network_guard.os.geteuid", return_value=1000),
            patch("network_guard.os.getegid", return_value=1000),
            patch("network_guard._interface_names", return_value=("lo",)),
            patch("network_guard._route_snapshot", return_value=((), ())),
            patch("network_guard._assert_external_connect_fails"),
            patch("network_guard.os.execvpe") as execvpe,
        ):
            result = child_main(args)
        self.assertEqual(result, 2)
        self.assertEqual(stderr.getvalue(), "network guard: network isolation deadline expired\n")
        self.assertNotIn("Traceback", stderr.getvalue())
        execvpe.assert_not_called()

    def test_external_connect_wait_is_bounded_and_rechecked(self) -> None:
        clock = [0.0]
        observed_timeout: list[float] = []

        class FakeSocket:
            def __enter__(self) -> "FakeSocket":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def settimeout(self, value: float) -> None:
                observed_timeout.append(value)

            def connect_ex(self, _address: tuple[str, int]) -> int:
                clock[0] = 2.0
                return 111

        with (
            patch("network_guard.time.monotonic", side_effect=lambda: clock[0]),
            patch("network_guard.socket.socket", return_value=FakeSocket()),
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "deadline expired"):
                _assert_external_connect_fails(deadline=1.0)
        self.assertEqual(observed_timeout, [0.25])


class ChildIsolationVerificationTests(unittest.TestCase):
    PARENT_NETNS = "net:[4026531840]"
    CHILD_NETNS = "net:[4026531841]"
    ZERO_CAPABILITIES = {"CapInh": 0, "CapPrm": 0, "CapEff": 0, "CapBnd": 0, "CapAmb": 0}

    def _assert_rejected(self, **overrides: object) -> None:
        values = {
            "child_netns": self.CHILD_NETNS,
            "euid": 1000,
            "egid": 1000,
            "capabilities": self.ZERO_CAPABILITIES,
            "no_new_privs": 1,
            "groups": (),
            "interfaces": ("lo",),
            "routes": ((), ()),
        }
        values.update(overrides)
        with (
            patch("network_guard.current_netns", return_value=values["child_netns"]),
            patch("network_guard.os.geteuid", return_value=values["euid"]),
            patch("network_guard.os.getegid", return_value=values["egid"]),
            patch("network_guard.process_security_state", return_value=(values["capabilities"], values["no_new_privs"])),
            patch("network_guard.os.getgroups", return_value=list(values["groups"])),
            patch("network_guard._interface_names", return_value=values["interfaces"]),
            patch("network_guard._route_snapshot", return_value=values["routes"]),
            patch("network_guard._assert_external_connect_fails"),
        ):
            with self.assertRaises(NetworkIsolationError):
                assert_isolated(
                    parent_netns=self.PARENT_NETNS,
                    expected_euid=1000,
                    expected_egid=1000,
                    require_no_capabilities=True,
                    address_space_limit_bytes=None,
                )

    def test_same_namespace_and_security_mismatches_are_rejected(self) -> None:
        self._assert_rejected(child_netns=self.PARENT_NETNS)
        for values in (
            {"euid": 1001},
            {"egid": 1001},
            {"capabilities": {**self.ZERO_CAPABILITIES, "CapEff": 1}},
            {"no_new_privs": 0},
            {"groups": (1001,)},
            {"interfaces": ("lo", "eth0")},
            {"routes": ((("eth0", "00000000"),), ())},
        ):
            with self.subTest(values=values):
                self._assert_rejected(**values)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("SLLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "SLLM_UNITTEST_COUNTS="
            + json.dumps({
                "collected": selected,
                "selected": selected,
                "passed": selected - failed - skipped,
                "failed": failed,
                "skipped": skipped,
                "deselected": 0,
            }, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
