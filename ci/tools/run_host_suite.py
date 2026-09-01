#!/usr/bin/env python3
"""Run one explicitly registered host row under fail-closed limits."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import resource
import selectors
import signal
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    command_content_hash,
    command_hash,
    fixture_size_bytes,
    identity,
    isolated_env,
    iso_z,
    load_manifests,
    manifest_bundle_hash,
    matrix_manifest_hash,
    registered_row_commands,
    result_report_bytes,
    sha256_bytes,
    sha256_json,
    toolchain_snapshot,
    tuple_digest,
    utc_now,
    validate_required_toolchain,
    validate_result_payload,
    worktree_status,
)
from network_guard import (  # noqa: E402
    NetworkIsolationError,
    prepare_isolation,
    verify_parent_restored,
    wrap_command,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INFRA = 2
EXIT_HARNESS = 3

COUNT_KEYS = ("collected", "selected", "passed", "failed", "skipped", "deselected")
EVIDENCE_MODE_REQUIRED_CI = "required-ci"
EVIDENCE_MODE_LOCAL_DEVELOPMENT = "local-development"
UNITTEST_WRAPPER_FLAG = "--_unittest-count-wrapper"
PROC_LIVE_STATES = frozenset({"R", "S", "D", "T", "t", "K", "W", "P", "I"})


class ProcessDeadlineExpired(ValueError):
    """The command deadline expired before the child process was launched."""


@dataclass(frozen=True)
class ProcessGroupMember:
    pid: int
    state: str
    rss_bytes: int


@dataclass(frozen=True)
class ProcessGroupSnapshot:
    members: tuple[ProcessGroupMember, ...]

    @property
    def live_members(self) -> tuple[int, ...]:
        return tuple(member.pid for member in self.members if member.state in PROC_LIVE_STATES)

    @property
    def live_rss_bytes(self) -> int:
        return sum(member.rss_bytes for member in self.members if member.state in PROC_LIVE_STATES)


def exact_arg(value: str, name: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError(f"{name} must be a 40-character lowercase SHA")
    return value


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", "--row-id", dest="row", choices=("h0", "h1", "h2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    parser.add_argument(
        "--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha",
        type=lambda value: exact_arg(value, "--expected-reviewed-sha"),
        default=os.environ.get("REVIEWED_SHA"),
    )
    parser.add_argument(
        "--tested-sha", "--expected-tested-sha", dest="tested_sha",
        type=lambda value: exact_arg(value, "--expected-tested-sha"),
        default=os.environ.get("TESTED_SHA"),
    )
    parser.add_argument(
        "--workflow-sha", "--expected-workflow-sha", dest="workflow_sha",
        type=lambda value: exact_arg(value, "--expected-workflow-sha"),
        default=os.environ.get("WORKFLOW_SHA"),
    )
    parser.add_argument(
        "--strict-ci", action="store_true",
        help="require a clean checkout, matching identities, and the pinned toolchain",
    )
    parser.add_argument(
        "--allow-dirty-local", action="store_true",
        help="explicitly permit a non-immutable local-development run",
    )
    parser.add_argument("--seed", type=int, help="must equal the versioned row seed")
    return parser


def fail_harness(message: str) -> int:
    print(f"harness error: {message}", file=sys.stderr)
    return EXIT_HARNESS


def validate_execution_identity(
    *,
    strict_ci: bool,
    allow_dirty_local: bool,
    worktree_clean: bool,
    head_sha: str,
    reviewed_sha: str,
    tested_sha: str,
    workflow_sha: str,
) -> str:
    if strict_ci and allow_dirty_local:
        raise ContractError("--allow-dirty-local is prohibited under --strict-ci")
    if strict_ci and not worktree_clean:
        raise ContractError("strict CI rejects a dirty worktree")
    if not strict_ci and not worktree_clean and not allow_dirty_local:
        raise ContractError("dirty local run requires the explicit --allow-dirty-local opt-out")
    for name, value in {
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
    }.items():
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ContractError(f"{name} must be a 40-character lowercase SHA")
        if value != head_sha:
            raise ContractError(f"{name} must exactly match checked-out HEAD {head_sha}")
    return EVIDENCE_MODE_REQUIRED_CI if strict_ci else EVIDENCE_MODE_LOCAL_DEVELOPMENT


def empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def is_cargo_test(argv: list[str]) -> bool:
    return bool(argv) and argv[0] == "cargo" and "test" in argv[1:]


def is_pytest(argv: list[str]) -> bool:
    return len(argv) >= 3 and argv[0] == sys.executable and argv[1:3] == ["-m", "pytest"]


def _is_unittest_command_shape(argv: list[str]) -> bool:
    if not argv or argv[0] != sys.executable or len(argv) < 2:
        return False
    return argv[1:3] == ["-m", "unittest"] or argv[1].endswith(".py")


def _is_registered_script(argv: list[str]) -> bool:
    if len(argv) != 2 or argv[0] != sys.executable:
        return False
    script = Path(argv[1])
    return (
        script.suffix == ".py"
        and not script.is_absolute()
        and ".." not in script.parts
        and len(script.parts) >= 3
        and script.parts[:2] == ("ci", "tests")
        and script.as_posix() == argv[1]
    )


def _registered_host_commands(repo: Path) -> list[list[str]]:
    suites, host, _ = load_manifests(repo)
    return [
        command
        for row in host["rows"]
        for _command_id, command in registered_row_commands(suites, row, repo)
    ]


def _registered_unittest_commands(repo: Path) -> list[list[str]]:
    return [
        command
        for command in _registered_host_commands(repo)
        if _is_registered_script(command) or command[1:3] == ["-m", "unittest"]
    ]


def _is_registered_unittest_invocation(argv: list[str], repo: Path) -> bool:
    return argv in _registered_unittest_commands(repo)


def _is_registered_direct_command(argv: list[str], repo: Path) -> bool:
    """Return true for an exact registered command that is not a unittest."""
    return any(
        command == argv and not _is_registered_unittest_invocation(command, repo)
        for command in _registered_host_commands(repo)
    )


def _is_unittest_candidate(argv: list[str], *, repo: Path = ROOT) -> bool:
    """Recognize unittest syntax while excluding registered direct commands."""
    return _is_unittest_command_shape(argv) and not _is_registered_direct_command(argv, repo)


def _validate_unittest_command(argv: list[str], repo: Path) -> None:
    if not _is_unittest_candidate(argv, repo=repo):
        return
    if _is_registered_unittest_invocation(argv, repo):
        return
    if argv[1:3] == ["-m", "unittest"]:
        raise ValueError("unittest discovery or selector list is not registered")
    script = Path(argv[1])
    if script.is_absolute() or ".." in script.parts:
        raise ValueError("unittest script path traversal or absolute path is prohibited")
    if len(script.parts) < 3 or script.parts[:2] != ("ci", "tests"):
        raise ValueError("unittest script must be under ci/tests")
    if script.as_posix() != argv[1]:
        raise ValueError("unittest script path is not canonical")
    raise ValueError("unittest script is not registered")


def is_unittest_script(argv: list[str], *, repo: Path = ROOT) -> bool:
    if not _is_unittest_candidate(argv, repo=repo):
        return False
    if argv[1:3] != ["-m", "unittest"] and not argv[1].startswith("ci/tests"):
        return False
    _validate_unittest_command(argv, repo)
    return True


def _machine_counts(output: str, *, prefix: str, source: str) -> tuple[dict[str, int], str | None, str]:
    records = re.findall(rf"(?m)^{re.escape(prefix)}=(\{{[^\r\n]+\}})\s*$", output)
    if len(records) != 1:
        return empty_counts(), f"expected exactly one {source} count record, got {len(records)}", source
    try:
        value = json.loads(records[0])
    except ValueError as exc:
        return empty_counts(), f"{source} count record is invalid JSON: {exc}", source
    if not isinstance(value, dict) or set(value) != set(COUNT_KEYS):
        return empty_counts(), f"{source} count record has unknown or missing keys", source
    counts: dict[str, int] = {}
    for key in COUNT_KEYS:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return empty_counts(), f"{source} count {key} is not a non-negative integer", source
        counts[key] = item
    if counts["collected"] != counts["selected"] + counts["deselected"]:
        return empty_counts(), f"{source} collected count is inconsistent", source
    if counts["selected"] != counts["passed"] + counts["failed"] + counts["skipped"]:
        return empty_counts(), f"{source} selected count is inconsistent", source
    warning = f"{source} reported zero tests selected" if counts["selected"] == 0 else None
    return counts, warning, source


def actual_counts(
    argv: list[str], output: str, exit_code: int, *, repo: Path = ROOT
) -> tuple[dict[str, int], str | None, str]:
    """Read framework outcomes; never substitute registry declarations."""
    if is_pytest(argv):
        return _machine_counts(output, prefix="SLLM_PYTEST_COUNTS", source="pytest-machine")
    if is_cargo_test(argv):
        counts = empty_counts()
        matches = re.findall(
            r"(?m)^test result: (?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored;\s+\d+ measured;\s+(\d+) filtered out;",
            output,
        )
        if not matches:
            return counts, "expected at least one Cargo test harness summary, got 0", "cargo-harness"
        for passed, failed, ignored, filtered in matches:
            counts["passed"] += int(passed)
            counts["failed"] += int(failed)
            # Rust's #[ignore] removes a test from the normal Cargo selection;
            # it is not a runtime skip of a selected host test.  Keep ignored
            # tests visible in collected/deselected accounting while retaining
            # the zero-selection fail-closed check below.
            counts["deselected"] += int(ignored) + int(filtered)
        counts["selected"] = counts["passed"] + counts["failed"] + counts["skipped"]
        counts["collected"] = counts["selected"] + counts["deselected"]
        warning = "Cargo test harnesses reported zero tests selected" if counts["selected"] == 0 else None
        return counts, warning, "cargo-harness"
    if is_unittest_script(argv, repo=repo):
        return _machine_counts(output, prefix="SLLM_UNITTEST_COUNTS", source="unittest-machine")
    return {
        "collected": 1,
        "selected": 1,
        "passed": 1 if exit_code == 0 else 0,
        "failed": 0 if exit_code == 0 else 1,
        "skipped": 0,
        "deselected": 0,
    }, None, "validator-command"


class _MachineCountResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.machine_outcomes: dict[str, str] = {}

    def _record(self, test: unittest.case.TestCase, outcome: str) -> None:
        priority = {"passed": 0, "skipped": 1, "failed": 2}
        previous = self.machine_outcomes.get(test.id())
        if previous is None or priority[outcome] > priority[previous]:
            self.machine_outcomes[test.id()] = outcome

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._record(test, "failed")
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._record(test, "failed")
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:  # noqa: N802
        self._record(test, "skipped")
        super().addSkip(test, reason)

    def addExpectedFailure(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._record(test, "skipped")
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._record(test, "failed")
        super().addUnexpectedSuccess(test)


def _unittest_ids(test: unittest.TestSuite | unittest.case.TestCase) -> list[str]:
    if isinstance(test, unittest.TestSuite):
        result: list[str] = []
        for child in test:
            result.extend(_unittest_ids(child))
        return result
    return [test.id()]


class _MachineCountRunner(unittest.TextTestRunner):
    resultclass = _MachineCountResult

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selected_ids: list[str] = []

    def run(self, test: unittest.TestSuite) -> _MachineCountResult:
        self.selected_ids = _unittest_ids(test)
        result = super().run(test)
        if not isinstance(result, _MachineCountResult):
            raise RuntimeError("unittest runner returned the wrong result class")
        return result


def run_unittest_count_wrapper(original_argv: list[str], *, repo: Path = ROOT) -> int:
    try:
        _validate_unittest_command(original_argv, repo)
    except ValueError as exc:
        print(f"unittest count wrapper: {exc}", file=sys.stderr)
        return 2
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    runner = _MachineCountRunner(verbosity=2)
    if original_argv[1:3] == ["-m", "unittest"]:
        program = unittest.main(
            module=None,
            argv=["python -m unittest", *original_argv[3:]],
            testRunner=runner,
            exit=False,
        )
        result = program.result
    elif len(original_argv) == 2:
        script = (repo / original_argv[1]).resolve(strict=True)
        suite = unittest.defaultTestLoader.discover(
            str(script.parent), pattern=script.name, top_level_dir=str(script.parent)
        )
        result = runner.run(suite)
    else:
        print("unittest count wrapper: unsupported command", file=sys.stderr)
        return 2
    if not isinstance(result, _MachineCountResult):
        print("unittest count wrapper: missing machine result", file=sys.stderr)
        return 2
    outcomes = {"passed": 0, "failed": 0, "skipped": 0}
    for test_id in runner.selected_ids:
        outcomes[result.machine_outcomes.get(test_id, "failed")] += 1
    counts = {
        "collected": len(runner.selected_ids),
        "selected": len(runner.selected_ids),
        **outcomes,
        "deselected": 0,
    }
    print("SLLM_UNITTEST_COUNTS=" + json.dumps(counts, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if result.wasSuccessful() else 1


def execution_argv(argv: list[str], *, repo: Path = ROOT) -> list[str]:
    if argv not in _registered_host_commands(repo):
        raise ValueError("command is not exactly registered in the host command allowlist")
    if _is_registered_unittest_invocation(argv, repo):
        return [sys.executable, str(Path(__file__).resolve()), UNITTEST_WRAPPER_FLAG, *argv[1:]]
    return argv


def row_output_breach(
    *,
    aggregate_output: int,
    max_row_output_bytes: int,
    commands_complete: bool,
    command_output_breach: bool = False,
) -> bool:
    """Apply strict row output accounting, including an incomplete exact boundary."""
    return (
        aggregate_output > max_row_output_bytes
        or command_output_breach
        or (not commands_complete and aggregate_output >= max_row_output_bytes)
    )


def signal_process_group(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def process_group_snapshot(pgid: int) -> ProcessGroupSnapshot:
    """Read one fail-closed procfs snapshot for PGID membership, state, and RSS."""
    members: list[ProcessGroupMember] = []
    proc_entries = list(Path("/proc").iterdir())
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise OSError(errno.EIO, "procfs page size is not positive")
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        # A process may exit or be replaced between ``iterdir`` and the stat
        # read.  Linux can expose a short/partial stat record for that narrow
        # window; retrying the same entry keeps the snapshot fail-closed for a
        # persistent malformed record while avoiding spurious INFRA_ERROR on a
        # transient procfs race during a busy Cargo build.
        stat: bytes | None = None
        fields: list[bytes] = []
        for attempt in range(3):
            try:
                candidate = (entry / "stat").read_bytes()
            except OSError as exc:
                if exc.errno in {errno.ENOENT, errno.ESRCH}:
                    stat = None
                    break
                raise OSError(exc.errno, f"cannot inspect {entry / 'stat'}: {exc.strerror}") from exc
            command_start = candidate.find(b"(")
            command_end = candidate.rfind(b")")
            candidate_fields = candidate[command_end + 2 :].split() if command_end >= 0 else []
            valid_state = len(candidate_fields) >= 22 and len(candidate_fields[0]) == 1
            valid = (
                command_start > 0
                and command_end > command_start
                and candidate[:command_start].strip() == entry.name.encode("ascii")
                and valid_state
                and candidate_fields[0].decode("ascii", "ignore") in PROC_LIVE_STATES | {"Z", "X", "x"}
                and candidate_fields[2].isdigit()
                and candidate_fields[21].isdigit()
            )
            if valid:
                stat = candidate
                fields = candidate_fields
                break
            if attempt == 2:
                raise OSError(errno.EPROTO, f"malformed procfs stat for PID {entry.name}")
        if stat is None:
            continue
        state = fields[0].decode("ascii")
        if int(fields[2]) == pgid:
            members.append(ProcessGroupMember(
                pid=int(entry.name),
                state=state,
                rss_bytes=int(fields[21]) * page_size,
            ))
    return ProcessGroupSnapshot(tuple(sorted(members, key=lambda member: member.pid)))


def live_process_group_members(pgid: int) -> tuple[int, ...]:
    """Return live Linux processes in *pgid*, excluding zombie artifacts."""
    return process_group_snapshot(pgid).live_members


def terminate_process_group(proc: subprocess.Popen[bytes]) -> tuple[tuple[int, ...], int]:
    """Terminate live group members and return any survivors plus peak group RSS."""
    peak_rss = 0
    inspection_error: BaseException | None = None
    live: tuple[int, ...] = ()
    try:
        snapshot = process_group_snapshot(proc.pid)
        peak_rss = max(peak_rss, snapshot.live_rss_bytes)
        live = snapshot.live_members
    except BaseException as exc:
        inspection_error = exc
    try:
        signal_process_group(proc, signal.SIGTERM)
        if inspection_error is None:
            term_deadline = time.monotonic() + 0.1
            while live and time.monotonic() < term_deadline:
                time.sleep(0.01)
                snapshot = process_group_snapshot(proc.pid)
                peak_rss = max(peak_rss, snapshot.live_rss_bytes)
                live = snapshot.live_members
    except BaseException as exc:
        if inspection_error is None:
            inspection_error = exc
    finally:
        try:
            # The unreaped group leader pins the PGID until wait4 below.
            signal_process_group(proc, signal.SIGKILL)
        except BaseException as exc:
            if inspection_error is None:
                inspection_error = exc
    if inspection_error is None:
        try:
            kill_deadline = time.monotonic() + 1.0
            snapshot = process_group_snapshot(proc.pid)
            peak_rss = max(peak_rss, snapshot.live_rss_bytes)
            live = snapshot.live_members
            while live and time.monotonic() < kill_deadline:
                time.sleep(0.01)
                snapshot = process_group_snapshot(proc.pid)
                peak_rss = max(peak_rss, snapshot.live_rss_bytes)
                live = snapshot.live_members
        except BaseException as exc:
            inspection_error = exc
    if inspection_error is not None:
        raise inspection_error
    return live, peak_rss


def process_tree_rss_bytes(root_pid: int) -> int:
    """Compatibility wrapper for callers that pass a process-group leader PID."""
    return process_group_snapshot(root_pid).live_rss_bytes


def runner_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def run_bounded_process(
    argv: list[str],
    *,
    repo: Path,
    timeout_seconds: float,
    max_rss_bytes: int,
    output_limit_bytes: int,
    deadline: float | None = None,
) -> tuple[bytes, bytes, int, float, bool, bool, bool, int, int, int, float, float]:
    """Execute one process group with bounded output, timeout, and RSS."""
    started = time.monotonic()
    if timeout_seconds <= 0 or max_rss_bytes <= 0 or output_limit_bytes <= 0:
        raise ValueError("process limits must be positive")
    if deadline is None:
        # Standalone callers have no outer command deadline; preserve their
        # historical timeout while run_command supplies its existing deadline.
        deadline = time.monotonic() + timeout_seconds
    command_env = isolated_env()
    command_env["SLLM_EMIT_TEST_COUNTS"] = "1"
    if time.monotonic() >= deadline:
        raise ProcessDeadlineExpired("process launch deadline expired")
    proc = subprocess.Popen(
        argv,
        cwd=repo,
        env=command_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector: selectors.BaseSelector | None = None
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    observed_output = {"stdout": 0, "stderr": 0}
    timed_out = output_breach = rss_breach = False
    observed_rss = 0
    cpu_user_seconds = cpu_system_seconds = 0.0
    returncode: int | None = None
    root_exited = False
    terminate_started: float | None = None
    kill_sent = False
    try:
        assert proc.stdout is not None and proc.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map() or not root_exited:
            now = time.monotonic()
            snapshot = process_group_snapshot(proc.pid)
            observed_rss = max(observed_rss, snapshot.live_rss_bytes)
            if observed_rss > max_rss_bytes:
                rss_breach = True
                if terminate_started is None:
                    terminate_started = now
                    signal_process_group(proc, signal.SIGTERM)
            if now >= deadline:
                timed_out = True
                if terminate_started is None:
                    terminate_started = now
                    signal_process_group(proc, signal.SIGTERM)
            if terminate_started is not None and not kill_sent and now - terminate_started >= 2.0:
                kill_sent = True
                signal_process_group(proc, signal.SIGKILL)
            if terminate_started is not None and now - terminate_started >= 4.0:
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            if selector.get_map():
                for key, _ in selector.select(0.02):
                    try:
                        data = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    except OSError as exc:
                        if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                            continue
                        raise
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    observed_output[key.data] += len(data)
                    used = len(captured["stdout"]) + len(captured["stderr"])
                    remaining = max(0, output_limit_bytes - used)
                    if len(data) > remaining:
                        if remaining:
                            captured[key.data].extend(data[:remaining])
                        output_breach = True
                        if terminate_started is None:
                            terminate_started = time.monotonic()
                            signal_process_group(proc, signal.SIGTERM)
                    else:
                        captured[key.data].extend(data)
            elif not root_exited:
                time.sleep(0.01)
            if not root_exited:
                try:
                    root_exited = os.waitid(
                        os.P_PID,
                        proc.pid,
                        os.WEXITED | os.WNOHANG | os.WNOWAIT,
                    ) is not None
                except ChildProcessError:
                    root_exited = True
    finally:
        cleanup_error: BaseException | None = None
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                cleanup_error = exc
        for stream in (proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        live_members: tuple[int, ...] = ()
        try:
            live_members, cleanup_rss = terminate_process_group(proc)
            observed_rss = max(observed_rss, cleanup_rss)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            try:
                signal_process_group(proc, signal.SIGKILL)
            except BaseException as kill_exc:
                if cleanup_error is None:
                    cleanup_error = kill_exc
        try:
            _, status, usage = os.wait4(proc.pid, 0)
            returncode = os.waitstatus_to_exitcode(status)
            proc.returncode = returncode
            observed_rss = max(
                observed_rss,
                int(usage.ru_maxrss) * 1024 if sys.platform.startswith("linux") else int(usage.ru_maxrss),
            )
            cpu_user_seconds = float(usage.ru_utime)
            cpu_system_seconds = float(usage.ru_stime)
        except ChildProcessError:
            returncode = proc.returncode if proc.returncode is not None else 127
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            returncode = proc.returncode if proc.returncode is not None else 127
        if live_members:
            cleanup_error = cleanup_error or OSError(
                "process-group cleanup retained live member(s): "
                + ",".join(str(pid) for pid in live_members)
            )
        if cleanup_error is not None:
            raise cleanup_error
    elapsed = time.monotonic() - started
    return (
        bytes(captured["stdout"]), bytes(captured["stderr"]), int(returncode), elapsed,
        timed_out, output_breach, rss_breach or observed_rss > max_rss_bytes,
        observed_rss, observed_output["stdout"], observed_output["stderr"],
        cpu_user_seconds, cpu_system_seconds,
    )


def run_command(
    command_id: str,
    argv: list[str],
    *,
    timeout_seconds: float,
    repo: Path,
    output_dir: Path,
    max_rss_bytes: int,
    output_limit_bytes: int,
    address_space_limit_bytes: int | None,
) -> tuple[dict[str, Any], str]:
    started = utc_now()
    command_started = time.monotonic()
    stdout = stderr = b""
    exit_code = 127
    timed_out = output_breach = rss_breach = False
    observed_rss = observed_stdout_bytes = observed_stderr_bytes = 0
    cpu_user_seconds = cpu_system_seconds = 0.0
    isolated = address_space_limit_enforced = process_launched = False
    strategy = "unavailable"
    state = "PASS"
    diagnostics: list[str] = []
    plan = None
    deadline = command_started + timeout_seconds
    launch_deadline_error: ProcessDeadlineExpired | None = None
    restoration_error: NetworkIsolationError | None = None
    try:
        plan = prepare_isolation(outer_deadline=deadline)
        strategy = plan.strategy
        remaining = timeout_seconds - (time.monotonic() - command_started)
        if remaining <= 0:
            state = "FAIL"
            timed_out = True
            diagnostics.append("network guard setup exhausted command wall time")
        else:
            wrapped = wrap_command(
                plan,
                execution_argv(argv, repo=repo),
                address_space_limit_bytes=address_space_limit_bytes,
                outer_deadline=deadline,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state = "FAIL"
                timed_out = True
                diagnostics.append("network guard setup exhausted command wall time")
            else:
                (
                    stdout, stderr, exit_code, _process_elapsed, timed_out,
                    output_breach, rss_breach, observed_rss, observed_stdout_bytes,
                    observed_stderr_bytes, cpu_user_seconds, cpu_system_seconds,
                ) = run_bounded_process(
                    wrapped,
                    repo=repo,
                    timeout_seconds=remaining,
                    max_rss_bytes=max_rss_bytes,
                    output_limit_bytes=output_limit_bytes,
                    deadline=deadline,
                )
                process_launched = True
    except ProcessDeadlineExpired as exc:
        launch_deadline_error = exc
    except NetworkIsolationError as exc:
        state = "INFRA_ERROR"
        diagnostics.append(f"network isolation failed: {exc}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        state = "INFRA_ERROR"
        diagnostics.append(f"cannot execute command: {exc}")
    finally:
        if plan is not None:
            try:
                # Restoration is mandatory even after the execution deadline.
                # A restored parent must preserve the command timeout as FAIL,
                # while an actual namespace mismatch remains INFRA_ERROR.
                verify_parent_restored(plan, outer_deadline=None)
            except NetworkIsolationError as exc:
                restoration_error = exc

    if restoration_error is not None:
        state = "INFRA_ERROR"
        diagnostics.insert(0, f"network isolation failed: {restoration_error}")
    elif launch_deadline_error is not None:
        state = "FAIL"
        timed_out = True
        diagnostics.append(str(launch_deadline_error))
    else:
        guard_rejected = exit_code == 2 and b"network guard:" in stderr
        isolated = process_launched and not guard_rejected
        address_space_limit_enforced = address_space_limit_bytes is not None and isolated

    elapsed = time.monotonic() - command_started
    combined = (stdout + b"\n" + stderr).decode("utf-8", "replace")
    try:
        counts, count_warning, count_source = actual_counts(argv, combined, exit_code, repo=repo)
    except (ContractError, ValueError) as exc:
        counts, count_warning, count_source = empty_counts(), str(exc), "count-validation"
    if not isolated:
        counts = empty_counts()
    if state == "PASS" and exit_code != 0:
        if exit_code == 2 and b"network guard:" in stderr:
            state = "INFRA_ERROR"
            diagnostics.append("network guard rejected isolated execution")
        else:
            state = "FAIL"
            diagnostics.append(f"command exited {exit_code}")
    if timed_out:
        state = "FAIL"
        diagnostics.append(f"command timeout after {timeout_seconds:.3f}s")
    if observed_stdout_bytes + observed_stderr_bytes > output_limit_bytes:
        output_breach = True
    if output_breach:
        state = "FAIL"
        diagnostics.append(f"command output exceeded {output_limit_bytes} bytes")
    if rss_breach or observed_rss > max_rss_bytes:
        state = "FAIL"
        diagnostics.append(f"max RSS {observed_rss} exceeds {max_rss_bytes} bytes")
    if elapsed > timeout_seconds:
        state = "FAIL"
        diagnostics.append(f"wall time {elapsed:.3f}s exceeds {timeout_seconds:.3f}s")
    if count_warning and state == "PASS":
        state = "FAIL"
        diagnostics.append(f"required test outcome is prohibited: {count_warning}")
    if counts["selected"] == 0 and state == "PASS":
        state = "FAIL"
        diagnostics.append("zero tests selected")
    if counts["failed"] and state == "PASS":
        state = "FAIL"
        diagnostics.append(f"required test command failed {counts['failed']} selected test(s)")
    if counts["skipped"] and state == "PASS":
        state = "FAIL"
        diagnostics.append(f"required test command skipped {counts['skipped']} selected test(s)")
    if restoration_error is not None:
        state = "INFRA_ERROR"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{command_id}.stdout").write_bytes(stdout)
    (output_dir / f"{command_id}.stderr").write_bytes(stderr)
    finished = utc_now()
    resource_record = {
        "wall_time_limit_seconds": round(timeout_seconds, 6),
        "timed_out": timed_out,
        "max_rss_bytes": observed_rss,
        "max_rss_limit_bytes": max_rss_bytes,
        "rss_breach": rss_breach or observed_rss > max_rss_bytes,
        "cpu_user_seconds": round(cpu_user_seconds, 6),
        "cpu_system_seconds": round(cpu_system_seconds, 6),
        "stdout_bytes": observed_stdout_bytes,
        "stderr_bytes": observed_stderr_bytes,
        "output_bytes": observed_stdout_bytes + observed_stderr_bytes,
        "stdout_captured_bytes": len(stdout),
        "stderr_captured_bytes": len(stderr),
        "captured_output_bytes": len(stdout) + len(stderr),
        "output_limit_bytes": output_limit_bytes,
        "output_breach": output_breach,
        "network_isolated": isolated,
        "network_guard_strategy": strategy,
        "address_space_limit_bytes": address_space_limit_bytes,
        "address_space_limit_enforced": address_space_limit_enforced,
    }
    step = {
        "step_id": command_id,
        "state": state,
        "started_at": iso_z(started),
        "finished_at": iso_z(finished),
        "duration_seconds": round(elapsed, 6),
        "exit_code": exit_code,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "diagnostic": "; ".join(diagnostics),
        "selection_required": True,
        "count_source": count_source,
        "counts": counts,
        "resource": resource_record,
    }
    return step, (stderr or stdout).decode("utf-8", "replace")[-2000:]


def write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_bytes = result_report_bytes(payload)
    report_path.write_bytes(report_bytes)
    (output_dir / "report.json.sha256").write_text(
        f"{sha256_bytes(report_bytes)}  {report_path.name}\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == UNITTEST_WRAPPER_FLAG:
        return run_unittest_count_wrapper(
            [sys.executable, *argv[1:]], repo=Path.cwd().resolve()
        )
    args = args_parser().parse_args(argv)
    try:
        repo = args.repo.resolve()
        output_dir = args.output_dir.resolve()
        suites, host, _ = load_manifests(repo)
        row = next(item for item in host["rows"] if item["row_id"] == args.row)
        if args.seed is not None and args.seed != row["seed"]:
            return fail_harness("seed override must equal the versioned row seed")
        if args.run_attempt < 1:
            return fail_harness("run attempt must be positive")
        if args.strict_ci and not all((args.reviewed_sha, args.tested_sha, args.workflow_sha)):
            return fail_harness("strict CI requires explicit reviewed/tested/workflow SHA values")
        initial_status = worktree_status(repo)
        initial_worktree_clean = not initial_status["tracked"] and not initial_status["untracked"]
        git_identity = identity(repo)
        reviewed_sha = args.reviewed_sha or git_identity["commit"]
        tested_sha = args.tested_sha or git_identity["commit"]
        workflow_sha = args.workflow_sha or git_identity["commit"]
        evidence_mode = validate_execution_identity(
            strict_ci=args.strict_ci,
            allow_dirty_local=args.allow_dirty_local,
            worktree_clean=initial_worktree_clean,
            head_sha=git_identity["commit"],
            reviewed_sha=reviewed_sha,
            tested_sha=tested_sha,
            workflow_sha=workflow_sha,
        )
        run_id = args.run_id or f"local-{git_identity['commit'][:12]}"
        if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
            return fail_harness("invalid run id")
        toolchain = toolchain_snapshot(repo)
        if args.strict_ci:
            validate_required_toolchain(
                toolchain,
                require_dev_rust=args.row in {"h0", "h1"},
                require_msrv_rust=args.row == "h0",
            )
        commands = registered_row_commands(suites, row, repo)
        fixture_bytes = fixture_size_bytes(repo)
        started = utc_now()
        started_monotonic = time.monotonic()
        steps: list[dict[str, Any]] = []
        diagnostics: list[str] = []
        row_output_exhausted = row_timeout_exhausted = False
        for command_id, command in commands:
            elapsed_before = time.monotonic() - started_monotonic
            remaining_wall = row["timeout_seconds"] - elapsed_before
            remaining_output = row["max_row_output_bytes"] - sum(
                step["resource"]["output_bytes"] for step in steps
            )
            if remaining_wall <= 0:
                row_timeout_exhausted = True
                diagnostics.append("row wall-time budget was exhausted before all commands ran")
                break
            if remaining_output <= 0:
                row_output_exhausted = True
                diagnostics.append("row output budget was exhausted before all commands ran")
                break
            step, detail = run_command(
                command_id,
                command,
                timeout_seconds=min(float(row["max_command_seconds"]), remaining_wall),
                repo=repo,
                output_dir=output_dir,
                max_rss_bytes=row["max_rss_bytes"],
                output_limit_bytes=min(row["max_command_output_bytes"], remaining_output),
                address_space_limit_bytes=row["address_space_limit_bytes"],
            )
            steps.append(step)
            if step["state"] != "PASS":
                diagnostics.append(f"{command_id}: {step['diagnostic']} {detail}".strip())
        final_status = worktree_status(repo)
        final_worktree_clean = not final_status["tracked"] and not final_status["untracked"]
        worktree_clean = initial_worktree_clean and final_worktree_clean
        post_execution_worktree_dirty = args.strict_ci and not final_worktree_clean
        if post_execution_worktree_dirty:
            dirty = final_status["tracked"] + final_status["untracked"]
            diagnostics.append("strict CI worktree became dirty during command execution: " + ", ".join(dirty[:8]))
        finished = utc_now()
        elapsed = time.monotonic() - started_monotonic
        counts = {key: sum(step["counts"][key] for step in steps) for key in COUNT_KEYS}
        cases = [
            {"case_id": step["step_id"], **{key: value for key, value in step.items() if key != "step_id"}}
            for step in steps
        ]
        aggregate_output = sum(step["resource"]["output_bytes"] for step in steps)
        aggregate_captured_output = sum(step["resource"]["captured_output_bytes"] for step in steps)
        command_rss = max((step["resource"]["max_rss_bytes"] for step in steps), default=0)
        runner_rss = runner_max_rss_bytes()
        aggregate_rss = max(command_rss, runner_rss)
        strategies = sorted({step["resource"]["network_guard_strategy"] for step in steps})
        commands_complete = len(steps) == len(commands)
        fixture_size_breach = fixture_bytes > row["fixture_size_limit_bytes"]
        output_breach = row_output_breach(
            aggregate_output=aggregate_output,
            max_row_output_bytes=row["max_row_output_bytes"],
            commands_complete=commands_complete,
            command_output_breach=any(step["resource"]["output_breach"] for step in steps),
        )
        rss_breach = aggregate_rss > row["max_rss_bytes"] or any(
            step["resource"]["rss_breach"] for step in steps
        )
        wall_time_breach = elapsed > row["timeout_seconds"] or row_timeout_exhausted
        row_resource = {
            "wall_time_limit_seconds": row["timeout_seconds"],
            "wall_time_breach": wall_time_breach,
            "max_rss_bytes": aggregate_rss,
            "max_rss_limit_bytes": row["max_rss_bytes"],
            "rss_breach": rss_breach,
            "runner_max_rss_bytes": runner_rss,
            "fixture_size_bytes": fixture_bytes,
            "fixture_size_limit_bytes": row["fixture_size_limit_bytes"],
            "fixture_size_breach": fixture_size_breach,
            "output_bytes": aggregate_output,
            "captured_output_bytes": aggregate_captured_output,
            "row_output_limit_bytes": row["max_row_output_bytes"],
            "output_breach": output_breach,
            "address_space_limit_bytes": row["address_space_limit_bytes"],
            "commands_expected": len(commands),
            "commands_executed": len(steps),
            "commands_complete": commands_complete,
            "network_isolated": bool(steps) and all(step["resource"]["network_isolated"] for step in steps),
            "network_guard_strategies": strategies or ["unavailable"],
        }
        if any(step["state"] == "INFRA_ERROR" for step in steps):
            state = "INFRA_ERROR"
        elif (
            any(step["state"] != "PASS" for step in steps)
            or row_timeout_exhausted
            or row_output_exhausted
            or fixture_size_breach
            or output_breach
            or rss_breach
            or wall_time_breach
            or not commands_complete
            or post_execution_worktree_dirty
        ):
            state = "FAIL"
        else:
            state = "PASS"
        if fixture_size_breach:
            diagnostics.append(f"fixture bytes {fixture_bytes} exceed {row['fixture_size_limit_bytes']} bytes")
        warnings = []
        if evidence_mode == EVIDENCE_MODE_LOCAL_DEVELOPMENT:
            warnings.append("LOCAL DEVELOPMENT ONLY: this report is not immutable evidence")
        if not worktree_clean:
            warnings.append("worktree was dirty before or after command execution")
        payload: dict[str, Any] = {
            "schema_version": "test-result-v1",
            "result_id": f"{args.row}.{run_id}.{args.run_attempt}",
            "suite_id": f"host-{args.row}",
            "tier": row["tier"],
            "state": state,
            "required": row["required"],
            "evidence_mode": evidence_mode,
            "run_id": run_id,
            "run_attempt": args.run_attempt,
            "reviewed_sha": reviewed_sha,
            "tested_sha": tested_sha,
            "workflow_sha": workflow_sha,
            "git_tree_oid": git_identity["tree"],
            "worktree_clean": worktree_clean,
            "matrix_manifest_sha256": matrix_manifest_hash(repo),
            "matrix_row_id": row["row_id"],
            "tuple_digest": tuple_digest(row),
            "command": [command for _, command in commands],
            "command_sha256": command_hash(command for _, command in commands),
            "toolchain": toolchain,
            "toolchain_sha256": sha256_json(toolchain),
            "artifact": {
                "content_sha256": command_content_hash(steps),
                "manifest_sha256": manifest_bundle_hash(repo),
            },
            "created_at": iso_z(started),
            "started_at": iso_z(started),
            "finished_at": iso_z(finished),
            "duration_seconds": round(elapsed, 6),
            "seed": row["seed"],
            "counts": counts,
            "resource": row_resource,
            "cases": cases,
            "steps": steps,
            "diagnostic": {
                "message": "all registered host commands passed" if state == "PASS" else "one or more required host gates failed",
                "errors": diagnostics,
                "warnings": warnings,
                "output_dir": str(output_dir),
                "network_disabled": True,
                "model_disabled": True,
                "gpu_fallback_disabled": True,
                "network_guard_self_test": bool(steps) and all(step["resource"]["network_isolated"] for step in steps),
            },
        }
        validate_result_payload(payload)
        write_result(output_dir, payload)
        immutable = evidence_mode == EVIDENCE_MODE_REQUIRED_CI and state == "PASS" and worktree_clean
        print(
            f"{args.row}: {state} collected={counts['collected']} selected={counts['selected']} "
            f"duration={elapsed:.3f}s evidence={evidence_mode} immutable={'true' if immutable else 'false'} output={output_dir}"
        )
        if state == "PASS":
            return EXIT_PASS
        if state == "INFRA_ERROR":
            return EXIT_INFRA
        return EXIT_FAIL
    except (ContractError, OSError, KeyError, StopIteration, ValueError) as exc:
        return fail_harness(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
