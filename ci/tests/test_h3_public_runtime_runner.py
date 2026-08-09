#!/usr/bin/env python3
"""Fake-tool tests for the H3 public-runtime runner boundary."""

from __future__ import annotations

import argparse
import errno
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import run_h3_public_runtime_compile as runner  # noqa: E402


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "strict_ci": True,
        "pinned_container": True,
        "observed_image_reference": runner.PINNED_IMAGE,
        "observed_image_config_digest": runner.PINNED_CONFIG,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_device_readobj(
    target: str = "gfx1030",
    *,
    flags: str | None = None,
    wave: int = 32,
    public_symbol: bool = False,
    duplicate_probe: bool = False,
    duplicate_dynamic: bool = False,
    duplicate_cuid: bool = False,
    cuid_name: str = "__hip_cuid_0123456789abcdef",
    cuid_binding: str = "Global (0x1)",
    cuid_type: str = "Object (0x1)",
    cuid_other: str = "0",
    cuid_section: str = ".bss",
    extra_symbol: str | None = None,
    extra_undefined_symbol: str | None = None,
    target_note: str | None = None,
    extra_target_note: str | None = None,
    omit_target: bool = False,
    omit_flags: bool = False,
    feature_lines: str = "",
) -> str:
    def symbol(name: str, binding: str, symbol_type: str, other: str, section: str) -> str:
        return "\n".join(
            [
                "  Symbol {",
                f"    Name: {name}",
                f"    Binding: {binding}",
                f"    Type: {symbol_type}",
                f"    Other: {other}",
                f"    Section: {section}",
                "  }",
            ]
        )

    probe = symbol("sllm_hip_compile_probe", "Global (0x1)", "Function (0x2)", "0", ".text")
    metadata = symbol("sllm_hip_compile_probe.kd", "Global (0x1)", "Object (0x1)", "0", ".rodata")
    cuid = symbol(cuid_name, cuid_binding, cuid_type, cuid_other, cuid_section)
    dynamic = symbol("_DYNAMIC", "Local (0x0)", "None (0x0)", "2", ".dynamic")
    device_symbols = [probe, metadata, cuid, dynamic]
    if duplicate_probe:
        device_symbols.append(probe)
    if duplicate_dynamic:
        device_symbols.append(dynamic)
    if duplicate_cuid:
        device_symbols.append(cuid)
    if public_symbol:
        device_symbols.append(symbol("sllm_context_create", "Global (0x1)", "Function (0x2)", "0", ".text"))
    if extra_symbol is not None:
        device_symbols.append(symbol(extra_symbol, "Global (0x1)", "Function (0x2)", "0", ".text"))
    if extra_undefined_symbol is not None:
        device_symbols.append(symbol(extra_undefined_symbol, "Global (0x1)", "None (0x0)", "0", "Undefined"))
    symbols = "\n".join(device_symbols)
    target_notes = [] if omit_target else [target_note or target]
    if extra_target_note is not None:
        target_notes.append(extra_target_note)
    flags_text = "" if omit_flags else f"  Flags [ (0x{(flags or runner.E_FLAGS[target])[2:]}) ]\n"
    notes_text = "\n".join(f"  amdhsa.target: amdgcn-amd-amdhsa--{note}" for note in target_notes)
    return f"""FileHeaders [
  Class: ELF64
  Arch: amdgcn
  ABIVersion: 4
{flags_text}]
]
Sections [
  Section {{
    Name: .text
    Size: 0x10
  }}
]
Notes [
{notes_text}
  .wavefront_size: {wave}
{feature_lines}
]
{symbols}
"""


class H3PublicRuntimeRunnerTests(unittest.TestCase):
    def test_runner_json_and_all_sidecars_never_follow_preplanted_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-public-publication-") as directory:
            root = Path(directory)
            output = root / "output"
            build = output / "build"
            build.mkdir(parents=True)
            artifacts = [
                build / "hip-compile-probe-gfx1030.o",
                build / "public-runtime-gfx1030.o",
                build / "rmsnorm-kernel-gfx1030.o",
                build / "rmsnorm-api-gfx1030.o",
                build / "public-runtime-gfx1030.elf",
                build / "probe-gfx1030.fatbin",
                build / "device-code-object-gfx1030.elf",
            ]
            for artifact in artifacts:
                artifact.write_bytes(b"compiler artifact")
            outside = root / "outside"
            outside.mkdir()
            target = outside / "target"
            target.write_bytes(b"external sentinel")

            for json_name in ("report.json", "hip-runtime-artifact.json"):
                with self.subTest(json_name=json_name):
                    json_path = output / json_name
                    json_path.symlink_to(target)
                    with self.assertRaises(runner.RuntimeContractError):
                        runner.write_json_with_sidecar(json_path, {"state": "FAIL"})
                    self.assertEqual(target.read_bytes(), b"external sentinel")
                    json_path.unlink()

            operations = [
                (output / "report.json.sha256", lambda: runner.write_json_with_sidecar(output / "report.json", {"state": "FAIL"})),
                (output / "hip-runtime-artifact.json.sha256", lambda: runner.write_json_with_sidecar(output / "hip-runtime-artifact.json", {"state": "FAIL"})),
            ]
            operations.extend((artifact.with_name(artifact.name + ".sha256"), lambda artifact=artifact: runner.digest_record(artifact)) for artifact in artifacts)
            for sidecar_path, operation in operations:
                with self.subTest(sidecar=sidecar_path.name):
                    sidecar_path.symlink_to(target)
                    with self.assertRaises(runner.RuntimeContractError):
                        operation()
                    self.assertEqual(target.read_bytes(), b"external sentinel")
                    sidecar_path.unlink()
                    for generated in (sidecar_path.with_name(sidecar_path.name.removesuffix(".sha256")),):
                        if generated.exists():
                            generated.unlink()

    def test_forced_failure_report_path_rejects_preplanted_report_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-public-failure-report-") as directory:
            root = Path(directory)
            repo = root / "repo"
            output = root / "output"
            outside = root / "outside"
            repo.mkdir()
            output.mkdir()
            outside.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "h3@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "h3-test"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "base", "-q"], check=True)
            (repo / "dirty.txt").write_text("dirty\n", encoding="ascii")
            target = outside / "target"
            target.write_bytes(b"external sentinel")
            report = output / "report.json"
            report.symlink_to(target)
            self.assertEqual(
                runner.main(
                    [
                        "--row",
                        "h3-public-gfx1030",
                        "--repo",
                        str(repo),
                        "--output-dir",
                        str(output),
                    ]
                ),
                1,
            )
            self.assertEqual(target.read_bytes(), b"external sentinel")
            self.assertTrue(report.is_symlink())

    def test_runner_publication_has_exact_success_set_and_cleans_forced_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-public-cleanup-") as directory:
            output = Path(directory) / "output"
            output.mkdir()
            runner.write_json_with_sidecar(output / "report.json", {"state": "PASS"})
            self.assertEqual({path.name for path in output.iterdir()}, {"report.json", "report.json.sha256"})

            forced = Path(directory) / "forced"
            forced.mkdir()
            real_link = runner._link_fd_no_replace

            def fail_after_sidecar_link(source_fd: int, directory_fd: int, name: str) -> None:
                real_link(source_fd, directory_fd, name)
                if name == "report.json.sha256":
                    raise runner.RuntimeContractError("forced post-link failure")

            with patch.object(runner, "_link_fd_no_replace", side_effect=fail_after_sidecar_link):
                with self.assertRaises(runner.RuntimeContractError):
                    runner.write_json_with_sidecar(forced / "report.json", {"state": "FAIL"})
            self.assertEqual(list(forced.iterdir()), [])

    def test_runner_output_leaf_and_ancestor_races_cannot_publish_external_files(self) -> None:
        for race in ("leaf", "ancestor"):
            with self.subTest(race=race), tempfile.TemporaryDirectory(prefix="sllm-h3-public-race-") as directory:
                root = Path(directory)
                output = root / "parent" / "output"
                output.mkdir(parents=True)
                outside = root / "outside"
                outside.mkdir()
                target = outside / "target"
                target.write_bytes(b"external sentinel")
                moved = root / f"moved-{race}"
                if race == "ancestor":
                    watched = output.parent
                else:
                    watched = output
                original_verify = runner._verify_directory_bindings
                calls = 0

                def swap_after_first_verify(bindings: list[runner._DirectoryBinding]) -> None:
                    nonlocal calls
                    original_verify(bindings)
                    calls += 1
                    if calls == 1:
                        if race == "ancestor":
                            watched.rename(moved)
                            watched.symlink_to(outside, target_is_directory=True)
                        else:
                            output.rename(moved)
                            output.symlink_to(outside, target_is_directory=True)

                try:
                    with patch.object(runner, "_verify_directory_bindings", side_effect=swap_after_first_verify):
                        with self.assertRaises(runner.RuntimeContractError):
                            runner.write_json_with_sidecar(output / "report.json", {"state": "FAIL"})
                    self.assertEqual(target.read_bytes(), b"external sentinel")
                    self.assertFalse((outside / "report.json").exists())
                    self.assertFalse((outside / "report.json.sha256").exists())
                finally:
                    if output.is_symlink():
                        output.unlink()
                    if output.parent.is_symlink():
                        output.parent.unlink()
                    if moved.exists():
                        if race == "ancestor":
                            moved.rename(root / "parent")
                        else:
                            moved.rename(output)

    def test_runner_build_leaf_and_ancestor_races_cannot_publish_external_sidecar(self) -> None:
        for race in ("leaf", "ancestor"):
            with self.subTest(race=race), tempfile.TemporaryDirectory(prefix="sllm-h3-build-race-") as directory:
                root = Path(directory)
                output = root / "parent" / "output"
                build = output / "build"
                build.mkdir(parents=True)
                artifact = build / "probe.o"
                artifact.write_bytes(b"compiler artifact")
                outside = root / "outside"
                outside.mkdir()
                target = outside / "target"
                target.write_bytes(b"external sentinel")
                moved = root / f"build-moved-{race}"
                output_path, output_fd, opened, output_bindings = runner._open_bound_directory(output, create_leaf=False)
                build_path, build_fd = runner._open_bound_child_directory(output_path, output_fd, "build", output_bindings)
                build_bindings = list(output_bindings)
                output_bindings = output_bindings[:-1]
                original_verify = runner._verify_directory_bindings
                calls = 0

                def swap_after_first_verify(bindings: list[runner._DirectoryBinding]) -> None:
                    nonlocal calls
                    original_verify(bindings)
                    calls += 1
                    if calls == 1:
                        if race == "leaf":
                            build.rename(moved)
                            build.symlink_to(outside, target_is_directory=True)
                        else:
                            output.rename(moved)
                            output.symlink_to(outside, target_is_directory=True)

                try:
                    with patch.object(runner, "_verify_directory_bindings", side_effect=swap_after_first_verify):
                        with self.assertRaises(runner.RuntimeContractError):
                            runner.digest_record(artifact, directory_fd=build_fd, bindings=build_bindings)
                    self.assertEqual(target.read_bytes(), b"external sentinel")
                    self.assertFalse((outside / "probe.o.sha256").exists())
                finally:
                    for fd in (build_fd, *reversed(opened)):
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    if build.is_symlink():
                        build.unlink()
                    if output.is_symlink():
                        output.unlink()
                    if moved.exists():
                        if race == "leaf":
                            moved.rename(build)
                        else:
                            moved.rename(output)

    def test_environment_requires_strict_pinned_image_and_network_namespace(self) -> None:
        for mutation in (
            {"strict_ci": False},
            {"pinned_container": False},
            {"observed_image_reference": "wrong"},
            {"observed_image_config_digest": "sha256:" + "0" * 64},
        ):
            with self.subTest(mutation=mutation):
                with patch.object(runner, "network_isolated", return_value=True), self.assertRaises(runner.RuntimeContractError):
                    runner.execution_environment(args(**mutation))
        with patch.object(runner, "network_isolated", return_value=False), self.assertRaises(runner.RuntimeContractError):
            runner.execution_environment(args())
        with patch.object(runner, "network_isolated", return_value=True):
            environment = runner.execution_environment(args())
        self.assertTrue(environment["network_isolated"])
        self.assertFalse(environment["pinned_container"] is False)

    def test_fake_argv_is_bounded_and_timeout_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-runner-") as directory:
            cwd = Path(directory)
            code, stdout, _stderr, _elapsed, timed_out, _rss = runner.run_argv(
                [sys.executable, "-c", "print('fake-tool')"],
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
                timeout=5,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout.strip(), b"fake-tool")
            self.assertFalse(timed_out)
            code, stdout, _stderr, _elapsed, timed_out, _rss = runner.run_argv(
                [sys.executable, "-c", "import sys; print(sys.argv[1])", "&&"],
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
                timeout=5,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout.strip(), b"&&")
            self.assertFalse(timed_out)
            code, _stdout, _stderr, _elapsed, timed_out, _rss = runner.run_argv(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
                timeout=0.05,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, 124)
            self.assertTrue(timed_out)

    def test_pidfd_preflight_fails_before_spawn_and_restores_runner_state(self) -> None:
        """Missing pidfd support must reject a full invocation before Popen."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-pidfd-preflight-") as directory:
            cwd = Path(directory)
            marker = cwd / "unexpected-child-marker"
            fd_before = len(os.listdir("/proc/self/fd"))
            subreaper_before = runner._child_subreaper_enabled()
            code = f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')"
            with (
                patch.object(os, "pidfd_open", side_effect=OSError(errno.ENOSYS, "pidfd disabled")),
                patch.object(subprocess, "Popen", wraps=subprocess.Popen) as popen,
                self.assertRaisesRegex(runner.RuntimeContractError, "runner preflight"),
            ):
                runner.run_argv(
                    [sys.executable, "-c", code],
                    cwd=cwd,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=1,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                )
            popen.assert_not_called()
            self.assertFalse(marker.exists(), "pidfd preflight spawned the command")
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
            self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)

        with patch.object(signal, "pidfd_send_signal", None), self.assertRaises(runner.RuntimeContractError):
            runner.run_argv(
                [sys.executable, "-c", "pass"],
                cwd=Path(tempfile.gettempdir()),
                env={"PATH": "/usr/bin:/bin"},
                timeout=1,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )

    def test_fake_argv_streams_combined_output_and_caps_flood(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-output-") as directory:
            code, stdout, stderr, _elapsed, timed_out, _rss = runner.run_argv(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.write('o' * 900); sys.stderr.write('e' * 900)",
                ],
                cwd=Path(directory),
                env={"PATH": "/usr/bin:/bin"},
                timeout=5,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, runner.OUTPUT_LIMIT_EXIT)
            self.assertFalse(timed_out)
            self.assertLessEqual(len(stdout) + len(stderr), 1024)

    def test_fake_argv_caps_sigterm_cleanup_flood_within_combined_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-term-flood-") as directory:
            handler = (
                "import signal,sys,time; "
                "signal.signal(signal.SIGTERM, lambda *_: (sys.stdout.write('o' * 4096), "
                "sys.stdout.flush(), sys.stderr.write('e' * 4096), sys.stderr.flush())); "
                "time.sleep(30)"
            )
            code, stdout, stderr, _elapsed, timed_out, _rss = runner.run_argv(
                [sys.executable, "-c", handler],
                cwd=Path(directory),
                env={"PATH": "/usr/bin:/bin"},
                timeout=0.05,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, 124)
            self.assertTrue(timed_out)
            self.assertLessEqual(len(stdout) + len(stderr), 1024)

    def test_fake_argv_enforces_live_tree_rss(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-rss-") as directory:
            child_code = "x=bytearray(64 * 1024 * 1024); [x.__setitem__(i, 1) for i in range(0, len(x), 4096)]; import time; time.sleep(2)"
            code, _stdout, _stderr, _elapsed, timed_out, rss = runner.run_argv(
                [
                    sys.executable,
                    "-c",
                    f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(2)",
                ],
                cwd=Path(directory),
                env={"PATH": "/usr/bin:/bin"},
                timeout=5,
                rss_limit=32 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(code, runner.RSS_LIMIT_EXIT)
            self.assertFalse(timed_out)
            self.assertGreater(rss, 32 * 1024 * 1024)

    def test_fake_argv_timeout_kills_and_cleans_descendant_process_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-timeout-") as directory:
            cwd = Path(directory)
            marker = cwd / "child.pid"
            child_code = "import os,signal,time; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            code = (
                "import subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(child.pid)); time.sleep(30)"
            )
            result = runner.run_argv(
                [sys.executable, "-c", code],
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
                timeout=0.1,
                rss_limit=256 * 1024 * 1024,
                output_limit=1024,
            )
            self.assertEqual(result[0], 124)
            self.assertTrue(result[4])
            child_pid = int(marker.read_text(encoding="ascii"))
            for _ in range(20):
                try:
                    state_line = Path(f"/proc/{child_pid}/stat").read_text(encoding="ascii")
                    right_paren = state_line.rfind(")")
                    state = state_line[right_paren + 2 :].split()[0]
                    if state == "Z":
                        self.fail(f"descendant process {child_pid} was left as a zombie")
                except ProcessLookupError:
                    break
                except FileNotFoundError:
                    break
                time.sleep(0.05)
            else:
                self.fail(f"descendant process {child_pid} survived timeout cleanup")

        # Interrupt-like BaseException subclasses must take the same
        # identity-safe cleanup path after binding, while preserving their
        # exact type/value for a caller that requested the interruption.
        for exception_type, value in ((KeyboardInterrupt, "injected interrupt"), (SystemExit, 73)):
            with self.subTest(exception=exception_type.__name__), tempfile.TemporaryDirectory(prefix="sllm-h3-base-exception-") as directory:
                cwd = Path(directory)
                marker = cwd / "root-child.pid"
                fd_before = len(os.listdir("/proc/self/fd"))
                subreaper_before = runner._child_subreaper_enabled()
                unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                root_pid: int | None = None
                child_pid: int | None = None
                injected = False
                waited_pids: list[int] = []
                real_live = runner._live_process_tree
                real_wait = subprocess.Popen.wait

                child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )

                def await_marker() -> tuple[int, int]:
                    deadline = time.monotonic() + 3
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                    self.assertTrue(marker.exists(), "bound root did not publish its child identity")
                    return tuple(int(item) for item in marker.read_text(encoding="ascii").split(":", 1))  # type: ignore[return-value]

                def interrupt_after_bind(*args: object, **kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
                    nonlocal injected, root_pid, child_pid
                    if not injected:
                        injected = True
                        root_pid, child_pid = await_marker()
                        raise exception_type(value)
                    return real_live(*args, **kwargs)

                def record_wait(process: subprocess.Popen[bytes], *args: object, **kwargs: object) -> int:
                    waited_pids.append(process.pid)
                    return real_wait(process, *args, **kwargs)

                def assert_absent(pid: int) -> None:
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        try:
                            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                        except FileNotFoundError:
                            return
                        right_paren = stat_text.rfind(")")
                        self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                        time.sleep(0.01)
                    self.fail(f"PID {pid} survived BaseException cleanup")

                try:
                    with (
                        patch.object(runner, "_live_process_tree", side_effect=interrupt_after_bind),
                        patch.object(subprocess.Popen, "wait", new=record_wait),
                        self.assertRaises(exception_type) as raised,
                    ):
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                    self.assertIs(type(raised.exception), exception_type)
                    if exception_type is SystemExit:
                        self.assertEqual(raised.exception.code, value)
                    else:
                        self.assertEqual(raised.exception.args, (value,))
                    self.assertIsNotNone(root_pid)
                    self.assertIsNotNone(child_pid)
                    assert root_pid is not None and child_pid is not None
                    assert_absent(root_pid)
                    assert_absent(child_pid)
                    self.assertIn(root_pid, waited_pids, "Popen.wait did not retain ownership of the bound root")
                    self.assertIsNone(unrelated.poll())
                    self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
                    self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
                finally:
                    for pid in (root_pid, child_pid):
                        if pid is None:
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass
                    unrelated.terminate()
                    unrelated.wait(timeout=3)

    def test_fake_argv_naturally_reaps_descendant_after_successful_root_exit(self) -> None:
        """A synchronized root/child handoff exercises the natural-drain path."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-natural-drain-") as directory:
            cwd = Path(directory)
            root_marker = cwd / "root.pid"
            child_marker = cwd / "child.exited"
            fd_before = len(os.listdir("/proc/self/fd"))
            release_read, release_write = os.pipe()
            outcome: list[object] = []
            natural_started = threading.Event()
            child_code = (
                "import os,sys; "
                "os.read(int(sys.argv[1]), 1); "
                f"open({str(child_marker)!r}, 'w').write(str(os.getpid()))"
            )
            root_code = (
                "import os,subprocess,sys; "
                f"release_fd={release_read}; "
                f"child_code={child_code!r}; "
                "child=subprocess.Popen([sys.executable, '-c', child_code, str(release_fd)], "
                "pass_fds=(release_fd,)); "
                f"open({str(root_marker)!r}, 'w').write(str(child.pid)); "
                "os.close(release_fd)"
            )

            def invoke() -> None:
                try:
                    outcome.append(
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                            pass_fds=(release_read,),
                        )
                    )
                except BaseException as exc:  # transfer the thread failure to the test thread
                    outcome.append(exc)

            real_natural_drain = runner._natural_drain_and_reap

            def observe_natural_drain(*args: object, **kwargs: object) -> tuple[bool, int, bool, bool]:
                natural_started.set()
                return real_natural_drain(*args, **kwargs)

            worker = threading.Thread(target=invoke, name="h3-natural-drain")
            with patch.object(runner, "_natural_drain_and_reap", side_effect=observe_natural_drain):
                worker.start()
                try:
                    self.assertTrue(natural_started.wait(timeout=5), "runner did not enter bounded natural drain")
                    os.close(release_write)
                    release_write = -1
                    worker.join(timeout=5)
                    self.assertFalse(worker.is_alive(), "natural-drain runner did not complete")
                    self.assertEqual(len(outcome), 1)
                    if isinstance(outcome[0], BaseException):
                        raise outcome[0]
                    result = outcome[0]
                    assert isinstance(result, tuple)
                    self.assertEqual(result[0], 0)
                    child_pid = int(root_marker.read_text(encoding="ascii"))
                    self.assertTrue(child_marker.is_file())
                    self.assertFalse(Path(f"/proc/{child_pid}").exists())
                finally:
                    if release_write >= 0:
                        os.close(release_write)
                    os.close(release_read)
                    worker.join(timeout=5)
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)

    def test_natural_drain_repeats_twenty_times_without_descendant_false_failure(self) -> None:
        """The root-zombie/reparent handoff remains stable across repeated runs."""

        fd_before = len(os.listdir("/proc/self/fd"))
        for iteration in range(20):
            with self.subTest(iteration=iteration), tempfile.TemporaryDirectory(prefix="sllm-h3-natural-repeat-") as directory:
                cwd = Path(directory)
                root_marker = cwd / "root.pid"
                child_marker = cwd / "child.exited"
                release_read, release_write = os.pipe()
                outcome: list[object] = []
                natural_started = threading.Event()
                natural_calls = 0
                child_code = (
                    "import os,sys; "
                    "os.read(int(sys.argv[1]), 1); "
                    f"open({str(child_marker)!r}, 'w').write(str(os.getpid()))"
                )
                root_code = (
                    "import os,subprocess,sys; "
                    f"release_fd={release_read}; child_code={child_code!r}; "
                    "child=subprocess.Popen([sys.executable, '-c', child_code, str(release_fd)], "
                    "pass_fds=(release_fd,)); "
                    f"open({str(root_marker)!r}, 'w').write(str(child.pid)); "
                    "os.close(release_fd)"
                )
                real_natural = runner._natural_drain_and_reap

                def observe_natural(*args: object, **kwargs: object) -> tuple[bool, int, bool, bool]:
                    nonlocal natural_calls
                    natural_calls += 1
                    natural_started.set()
                    return real_natural(*args, **kwargs)

                def invoke() -> None:
                    try:
                        outcome.append(
                            runner.run_argv(
                                [sys.executable, "-c", root_code],
                                cwd=cwd,
                                env={"PATH": "/usr/bin:/bin"},
                                timeout=5,
                                rss_limit=256 * 1024 * 1024,
                                output_limit=1024,
                                pass_fds=(release_read,),
                            )
                        )
                    except BaseException as exc:
                        outcome.append(exc)

                worker = threading.Thread(target=invoke, name=f"h3-natural-repeat-{iteration}")
                try:
                    with patch.object(runner, "_natural_drain_and_reap", side_effect=observe_natural):
                        worker.start()
                        self.assertTrue(natural_started.wait(timeout=5), "runner did not enter repeated natural drain")
                        os.close(release_write)
                        release_write = -1
                        worker.join(timeout=5)
                    self.assertFalse(worker.is_alive(), "repeated natural-drain runner did not complete")
                    self.assertEqual(len(outcome), 1)
                    if isinstance(outcome[0], BaseException):
                        raise outcome[0]
                    result = outcome[0]
                    assert isinstance(result, tuple)
                    self.assertEqual(result[0], 0)
                    self.assertFalse(result[4])
                    self.assertEqual(natural_calls, 1)
                    child_pid = int(root_marker.read_text(encoding="ascii"))
                    self.assertTrue(child_marker.is_file())
                    self.assertFalse(Path(f"/proc/{child_pid}").exists())
                finally:
                    if release_write >= 0:
                        os.close(release_write)
                    os.close(release_read)
                    worker.join(timeout=5)
        self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)

    def test_fake_argv_successful_root_with_lingering_descendant_returns_descendant_exit(self) -> None:
        """A child blocked on a held pipe must cross the bounded cleanup path."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-lingering-descendant-") as directory:
            cwd = Path(directory)
            marker = cwd / "child.pid"
            fd_before = len(os.listdir("/proc/self/fd"))
            release_read, release_write = os.pipe()
            child_code = (
                "import os,signal,sys; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "os.read(int(sys.argv[1]), 1)"
            )
            root_code = (
                "import os,subprocess,sys; "
                f"release_fd={release_read}; "
                f"child_code={child_code!r}; "
                "child=subprocess.Popen([sys.executable, '-c', child_code, str(release_fd)], "
                "pass_fds=(release_fd,)); "
                f"open({str(marker)!r}, 'w').write(str(child.pid)); "
                "os.close(release_fd)"
            )
            try:
                result = runner.run_argv(
                    [sys.executable, "-c", root_code],
                    cwd=cwd,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=5,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                    pass_fds=(release_read,),
                )
            finally:
                os.close(release_read)
                os.close(release_write)
            self.assertEqual(result[0], runner.DESCENDANT_EXIT)
            child_pid = int(marker.read_text(encoding="ascii"))
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)

    def test_root_identity_bind_failure_reaps_unbound_private_session(self) -> None:
        """A failed bind cannot leave an immediate-fork private session behind."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-unbound-bind-") as directory:
            cwd = Path(directory)
            root_marker = cwd / "root-child.pid"
            child_marker = cwd / "child-ready.pid"
            hold_read, hold_write = os.pipe()
            fd_before = len(os.listdir("/proc/self/fd"))
            subreaper_before = runner._child_subreaper_enabled()
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            root_pid: int | None = None
            child_pid: int | None = None

            def wait_for(path: Path) -> None:
                deadline = time.monotonic() + 3
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(path.exists(), f"missing synchronization marker {path.name}")

            def assert_absent(pid: int) -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                    except FileNotFoundError:
                        return
                    right_paren = stat_text.rfind(")")
                    self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                    time.sleep(0.01)
                self.fail(f"PID {pid} survived unbound private-session cleanup")

            child_code = (
                "import os,signal,sys; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(child_marker)!r}, 'w').write(str(os.getpid())); "
                "os.read(int(sys.argv[1]), 1)"
            )
            root_code = (
                "import os,subprocess,sys; "
                f"hold_fd={hold_read}; child_code={child_code!r}; "
                "child=subprocess.Popen([sys.executable, '-c', child_code, str(hold_fd)], pass_fds=(hold_fd,)); "
                f"open({str(root_marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                "os.close(hold_fd)"
            )
            real_bind = runner._root_identity_after_spawn

            def bind_then_fail(pid: int, baseline: set[runner.ProcessIdentity] | frozenset[runner.ProcessIdentity]) -> runner.ProcessIdentity:
                nonlocal root_pid, child_pid
                identity = real_bind(pid, baseline)
                wait_for(root_marker)
                wait_for(child_marker)
                root_pid, child_pid = (int(value) for value in root_marker.read_text(encoding="ascii").split(":", 1))
                self.assertEqual(root_pid, pid)
                self.assertEqual(identity[0], pid)
                self.assertEqual(os.getpgid(root_pid), root_pid)
                self.assertEqual(os.getpgid(child_pid), root_pid)
                raise runner.RuntimeContractError("injected root identity binding failure")

            try:
                with patch.object(runner, "_root_identity_after_spawn", side_effect=bind_then_fail):
                    with self.assertRaisesRegex(runner.RuntimeContractError, "injected root identity binding failure"):
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                            pass_fds=(hold_read,),
                        )
                assert root_pid is not None and child_pid is not None
                assert_absent(root_pid)
                assert_absent(child_pid)
                self.assertIsNone(unrelated.poll())
                self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                for fd in (hold_read, hold_write):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                for pid in (root_pid, child_pid):
                    if pid is None:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_root_bind_baseexception_preserves_exact_original_when_cleanup_fails(self) -> None:
        """Bind interrupts keep their exact value while cleanup diagnostics are bounded notes."""

        fd_before = len(os.listdir("/proc/self/fd"))
        subreaper_before = runner._child_subreaper_enabled()
        for original in (KeyboardInterrupt("injected bind interrupt"), SystemExit(73)):
            with self.subTest(exception=type(original).__name__), tempfile.TemporaryDirectory(prefix="sllm-h3-bind-baseexception-") as directory:
                cwd = Path(directory)
                marker = cwd / "root-child.pid"
                unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                root_pid: int | None = None
                child_pid: int | None = None
                real_bind = runner._root_identity_after_spawn
                real_cleanup = runner._cleanup_unbound_private_session
                child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )

                def wait_for_marker() -> None:
                    deadline = time.monotonic() + 3
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                    self.assertTrue(marker.exists(), "bind BaseException root did not publish identities")

                def bind_then_raise(pid: int, baseline: set[runner.ProcessIdentity] | frozenset[runner.ProcessIdentity]) -> runner.ProcessIdentity:
                    nonlocal root_pid, child_pid
                    identity = real_bind(pid, baseline)
                    wait_for_marker()
                    root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                    self.assertEqual(root_pid, pid)
                    self.assertEqual(identity[0], pid)
                    raise original

                def cleanup_then_fail(process: subprocess.Popen[bytes], selector: selectors.BaseSelector) -> None:
                    real_cleanup(process, selector)
                    raise runner.RuntimeContractError("injected unbound cleanup failure")

                def assert_absent(pid: int) -> None:
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        try:
                            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                        except FileNotFoundError:
                            return
                        right_paren = stat_text.rfind(")")
                        self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                        time.sleep(0.01)
                    self.fail(f"PID {pid} survived BaseException bind cleanup")

                try:
                    with (
                        patch.object(runner, "_root_identity_after_spawn", side_effect=bind_then_raise),
                        patch.object(runner, "_cleanup_unbound_private_session", side_effect=cleanup_then_fail),
                        self.assertRaises(BaseException) as raised,
                    ):
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                    self.assertIs(raised.exception, original)
                    self.assertIs(type(raised.exception), type(original))
                    self.assertEqual(raised.exception.args, original.args)
                    notes = "\n".join(getattr(raised.exception, "__notes__", []))
                    self.assertIn("unbound private-session cleanup failed", notes)
                    self.assertIn("injected unbound cleanup failure", notes)
                    assert root_pid is not None and child_pid is not None
                    assert_absent(root_pid)
                    assert_absent(child_pid)
                    self.assertIsNone(unrelated.poll())
                finally:
                    for pid in (root_pid, child_pid):
                        if pid is None:
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass
                    unrelated.terminate()
                    unrelated.wait(timeout=3)
        self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
        self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)

    def test_bound_spawn_oserror_cleanup_failed_escalates_to_emergency(self) -> None:
        """A bound OSError returns only after emergency cleanup proves removal."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-bound-oserror-emergency-") as directory:
            cwd = Path(directory)
            marker = cwd / "root-child.pid"
            fd_before = len(os.listdir("/proc/self/fd"))
            subreaper_before = runner._child_subreaper_enabled()
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            root_pid: int | None = None
            child_pid: int | None = None
            status_calls = 0
            terminate_calls = 0
            emergency_calls = 0
            original_error = OSError("injected bound operation failure")
            real_emergency = runner._emergency_cleanup_after_observation_failure
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            root_code = (
                "import os,subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                "time.sleep(30)"
            )

            def fail_after_bind(*args: object, **kwargs: object) -> int | None:
                nonlocal status_calls, root_pid, child_pid
                status_calls += 1
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(marker.exists(), "bound OSError root did not publish identities")
                root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                raise original_error

            def terminate_reports_unproven(*args: object, **kwargs: object) -> tuple[int, int, bool]:
                nonlocal terminate_calls
                terminate_calls += 1
                return 127, 0, True

            def observe_emergency(*args: object, **kwargs: object) -> bool:
                nonlocal emergency_calls
                emergency_calls += 1
                return real_emergency(*args, **kwargs)

            def assert_absent(pid: int) -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                    except FileNotFoundError:
                        return
                    right_paren = stat_text.rfind(")")
                    self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                    time.sleep(0.01)
                self.fail(f"PID {pid} survived bound OSError emergency cleanup")

            try:
                with (
                    patch.object(runner, "_root_exit_status_without_reap", side_effect=fail_after_bind),
                    patch.object(runner, "_terminate_and_reap", side_effect=terminate_reports_unproven),
                    patch.object(runner, "_emergency_cleanup_after_observation_failure", side_effect=observe_emergency),
                ):
                    result = runner.run_argv(
                        [sys.executable, "-c", root_code],
                        cwd=cwd,
                        env={"PATH": "/usr/bin:/bin"},
                        timeout=5,
                        rss_limit=256 * 1024 * 1024,
                        output_limit=1024,
                    )
                self.assertEqual(status_calls, 1)
                self.assertEqual(terminate_calls, 1)
                self.assertEqual(emergency_calls, 1)
                self.assertEqual(result[0], runner.DESCENDANT_EXIT)
                self.assertIn(b"injected bound operation failure", result[2])
                self.assertIn(b"bounded emergency cleanup proved audited process removal", result[2])
                assert root_pid is not None and child_pid is not None
                assert_absent(root_pid)
                assert_absent(child_pid)
                self.assertIsNone(unrelated.poll())
                self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                for pid in (root_pid, child_pid):
                    if pid is None:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_bound_spawn_cleanup_unproven_is_explicit_repeated_and_deadline_bounded(self) -> None:
        """Repeated OSError/SubprocessError cleanup failures never become ordinary 127."""

        fd_before = len(os.listdir("/proc/self/fd"))
        subreaper_before = runner._child_subreaper_enabled()
        for error_type, error_text in ((OSError, "injected bound OSError"), (subprocess.SubprocessError, "injected bound subprocess error")):
            for iteration in range(3):
                with self.subTest(error=error_type.__name__, iteration=iteration), tempfile.TemporaryDirectory(prefix="sllm-h3-bound-cleanup-unproven-") as directory:
                    cwd = Path(directory)
                    marker = cwd / "root-child.pid"
                    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                    root_pid: int | None = None
                    child_pid: int | None = None
                    scan_starts: list[float] = []
                    original_error = error_type(error_text)
                    original_snapshot = runner._private_cleanup_snapshot
                    real_emergency = runner._emergency_cleanup_after_observation_failure
                    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
                    root_code = (
                        "import os,subprocess,sys,time; "
                        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                        f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                        "time.sleep(30)"
                    )

                    def fail_after_bind(*args: object, **kwargs: object) -> int | None:
                        nonlocal root_pid, child_pid
                        deadline = time.monotonic() + 3
                        while not marker.exists() and time.monotonic() < deadline:
                            time.sleep(0.005)
                        self.assertTrue(marker.exists(), "unproven-cleanup root did not publish identities")
                        root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                        raise original_error

                    def terminate_reports_unproven(*args: object, **kwargs: object) -> tuple[int, int, bool]:
                        return 127, 0, True

                    def delayed_snapshot() -> dict[int, tuple[int, int, int, int]]:
                        scan_starts.append(time.monotonic())
                        time.sleep(0.25)
                        return original_snapshot()

                    def emergency_then_unprove(*args: object, **kwargs: object) -> bool:
                        real_emergency(*args, **kwargs)
                        return False

                    def assert_absent(pid: int) -> None:
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            try:
                                stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                            except FileNotFoundError:
                                return
                            right_paren = stat_text.rfind(")")
                            self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                            time.sleep(0.01)
                        self.fail(f"PID {pid} survived repeated unproven cleanup")

                    started = time.monotonic()
                    try:
                        with (
                            patch.object(runner, "_root_exit_status_without_reap", side_effect=fail_after_bind),
                            patch.object(runner, "_terminate_and_reap", side_effect=terminate_reports_unproven),
                            patch.object(runner, "_private_cleanup_snapshot", side_effect=delayed_snapshot),
                            patch.object(runner, "_EMERGENCY_CLEANUP_SECONDS", 0.05),
                            patch.object(runner, "_POST_DEADLINE_REAP_SECONDS", 0.05),
                            patch.object(runner, "_emergency_cleanup_after_observation_failure", side_effect=emergency_then_unprove),
                            self.assertRaises(runner.RuntimeContractError) as raised,
                        ):
                            runner.run_argv(
                                [sys.executable, "-c", root_code],
                                cwd=cwd,
                                env={"PATH": "/usr/bin:/bin"},
                                timeout=5,
                                rss_limit=256 * 1024 * 1024,
                                output_limit=1024,
                            )
                        self.assertIn("bound process cleanup-unproven", str(raised.exception))
                        self.assertIn(error_text, str(raised.exception))
                        self.assertIs(raised.exception.__cause__, original_error)
                        self.assertEqual(len(scan_starts), 1, "emergency cleanup started a post-deadline full scan")
                        self.assertLess(time.monotonic() - started, 0.80)
                        assert root_pid is not None and child_pid is not None
                        assert_absent(root_pid)
                        assert_absent(child_pid)
                        self.assertIsNone(unrelated.poll())
                        self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
                        self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
                    finally:
                        for pid in (root_pid, child_pid):
                            if pid is None:
                                continue
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            try:
                                os.waitpid(pid, 0)
                            except ChildProcessError:
                                pass
                        unrelated.terminate()
                        unrelated.wait(timeout=3)
        self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
        self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)

    # Both fallback cleanup paths use the same rule: one scan already in
        # progress may overrun a short deadline, but no post-deadline scan is
        # allowed and the outcome remains fail-closed.
        for cleanup_kind in ("emergency", "unbound"):
            with self.subTest(cleanup=cleanup_kind), tempfile.TemporaryDirectory(prefix=f"sllm-h3-{cleanup_kind}-deadline-") as directory:
                cwd = Path(directory)
                marker = cwd / "root-child.pid"
                root_pid: int | None = None
                child_pid: int | None = None
                selector = selectors.DefaultSelector()
                process: subprocess.Popen[bytes] | None = None
                scan_starts: list[float] = []
                original_snapshot = runner._private_cleanup_snapshot
                child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )

                def delayed_snapshot() -> dict[int, tuple[int, int, int, int]]:
                    scan_starts.append(time.monotonic())
                    time.sleep(0.25)
                    return original_snapshot()

                def assert_absent(pid: int) -> None:
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        try:
                            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                        except FileNotFoundError:
                            return
                        right_paren = stat_text.rfind(")")
                        self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                        time.sleep(0.01)
                    self.fail(f"PID {pid} survived {cleanup_kind} deadline cleanup")

                try:
                    with runner._process_observation_scope() as baseline:
                        process = subprocess.Popen(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            start_new_session=True,
                            preexec_fn=runner._install_child_containment,
                        )
                        root_identity = runner._root_identity_after_spawn(process.pid, baseline)
                        assert process.stdout is not None and process.stderr is not None
                        for stream, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                            os.set_blocking(pipe.fileno(), False)
                            selector.register(pipe.fileno(), selectors.EVENT_READ, stream)
                        deadline = time.monotonic() + 3
                        while not marker.exists() and time.monotonic() < deadline:
                            time.sleep(0.005)
                        self.assertTrue(marker.exists(), f"{cleanup_kind} root did not publish identities")
                        root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                        started = time.monotonic()
                        with (
                            patch.object(runner, "_UNBOUND_PRIVATE_CLEANUP_SECONDS", 0.05),
                            patch.object(runner, "_EMERGENCY_CLEANUP_SECONDS", 0.05),
                            patch.object(runner, "_private_cleanup_snapshot", side_effect=delayed_snapshot),
                        ):
                            if cleanup_kind == "emergency":
                                cleanup_ok = runner._emergency_cleanup_after_observation_failure(
                                    process,
                                    selector,
                                    {"stdout": bytearray(), "stderr": bytearray()},
                                    1024,
                                    0,
                                    root_identity,
                                    {root_identity},
                                    baseline,
                                )
                                self.assertFalse(cleanup_ok)
                            else:
                                with self.assertRaisesRegex(runner.RuntimeContractError, "could not prove"):
                                    runner._cleanup_unbound_private_session(process, selector)
                        self.assertEqual(len(scan_starts), 1)
                        self.assertLess(scan_starts[0], started + 0.05)
                        self.assertLess(time.monotonic() - started, 0.80)
                        assert root_pid is not None and child_pid is not None
                        assert_absent(root_pid)
                        assert_absent(child_pid)
                finally:
                    if process is not None:
                        runner._close_process_pipes(process)
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=1)
                        except (OSError, subprocess.SubprocessError):
                            pass
                    runner._close_streams(selector)
                    for pid in (root_pid, child_pid):
                        if pid is None:
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass

    def test_emergency_and_unbound_cleanup_each_repeat_twenty_times(self) -> None:
        """Deadline-crossing cleanup reaps known children in every repetition."""

        fd_before = len(os.listdir("/proc/self/fd"))
        subreaper_before = runner._child_subreaper_enabled()

        def assert_absent(pid: int, label: str) -> None:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                except FileNotFoundError:
                    return
                right_paren = stat_text.rfind(")")
                self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"{label} PID {pid} remained a zombie")
                time.sleep(0.01)
            self.fail(f"{label} PID {pid} survived repeated cleanup")

        def run_cleanup(cleanup_kind: str, iteration: int) -> None:
            with tempfile.TemporaryDirectory(prefix=f"sllm-h3-{cleanup_kind}-repeat-") as directory:
                cwd = Path(directory)
                marker = cwd / "root-child.pid"
                unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                selector = selectors.DefaultSelector()
                process: subprocess.Popen[bytes] | None = None
                root_pid: int | None = None
                child_pid: int | None = None
                scan_starts: list[float] = []
                original_snapshot = runner._private_cleanup_snapshot
                child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )

                def delayed_snapshot() -> dict[int, tuple[int, int, int, int]]:
                    scan_starts.append(time.monotonic())
                    time.sleep(0.25)
                    return original_snapshot()

                try:
                    with runner._process_observation_scope() as baseline:
                        process = subprocess.Popen(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            start_new_session=True,
                            preexec_fn=runner._install_child_containment,
                        )
                        root_identity = runner._root_identity_after_spawn(process.pid, baseline)
                        assert process.stdout is not None and process.stderr is not None
                        for stream, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                            os.set_blocking(pipe.fileno(), False)
                            selector.register(pipe.fileno(), selectors.EVENT_READ, stream)
                        deadline = time.monotonic() + 3
                        marker_values: tuple[int, int] | None = None
                        while marker_values is None and time.monotonic() < deadline:
                            try:
                                raw_marker = marker.read_text(encoding="ascii")
                                left, right = raw_marker.split(":", 1)
                                if left and right:
                                    marker_values = int(left), int(right)
                            except (FileNotFoundError, ValueError):
                                pass
                            time.sleep(0.005)
                        self.assertIsNotNone(marker_values, f"{cleanup_kind} repeat {iteration} did not publish identities")
                        assert marker_values is not None
                        root_pid, child_pid = marker_values
                        started = time.monotonic()
                        with (
                            patch.object(runner, "_UNBOUND_PRIVATE_CLEANUP_SECONDS", 0.05),
                            patch.object(runner, "_EMERGENCY_CLEANUP_SECONDS", 0.05),
                            patch.object(runner, "_private_cleanup_snapshot", side_effect=delayed_snapshot),
                        ):
                            if cleanup_kind == "emergency":
                                cleanup_ok = runner._emergency_cleanup_after_observation_failure(
                                    process,
                                    selector,
                                    {"stdout": bytearray(), "stderr": bytearray()},
                                    1024,
                                    0,
                                    root_identity,
                                    {root_identity},
                                    baseline,
                                )
                                self.assertFalse(cleanup_ok)
                            else:
                                with self.assertRaisesRegex(runner.RuntimeContractError, "could not prove"):
                                    runner._cleanup_unbound_private_session(process, selector)
                        self.assertEqual(len(scan_starts), 1)
                        self.assertLess(scan_starts[0], started + 0.05)
                        self.assertLess(time.monotonic() - started, 0.80)
                        assert root_pid is not None and child_pid is not None
                        assert_absent(root_pid, cleanup_kind)
                        assert_absent(child_pid, cleanup_kind)
                        self.assertIsNone(unrelated.poll())
                finally:
                    if process is not None:
                        runner._close_process_pipes(process)
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=1)
                        except (OSError, subprocess.SubprocessError):
                            pass
                    runner._close_streams(selector)
                    for pid in (root_pid, child_pid):
                        if pid is None:
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass
                    unrelated.terminate()
                    unrelated.wait(timeout=3)

        for cleanup_kind in ("emergency", "unbound"):
            for iteration in range(20):
                with self.subTest(cleanup=cleanup_kind, iteration=iteration):
                    run_cleanup(cleanup_kind, iteration)
        self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
        self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)

    def test_process_tree_and_operations_reject_start_time_mismatches(self) -> None:
        root_identity = (41001, 101)
        child_identity = (41002, 102)
        adopted_identity = (41003, 103)
        snapshot = {
            root_identity[0]: (1, root_identity[0], 4096, root_identity[1]),
            child_identity[0]: (root_identity[0], root_identity[0], 8192, child_identity[1]),
            adopted_identity[0]: (os.getpid(), 41003, 16384, adopted_identity[1]),
        }
        with patch.object(runner, "_proc_snapshot", return_value=snapshot):
            tree, rss = runner._live_process_tree(root_identity, {root_identity}, set())
        self.assertEqual(tree, {root_identity, child_identity})
        self.assertEqual(rss, 4096 + 8192)

        root_reused = dict(snapshot)
        root_reused[root_identity[0]] = (1, root_identity[0], 4096, 999)
        with patch.object(runner, "_proc_snapshot", return_value=root_reused):
            tree, rss = runner._live_process_tree(root_identity, {root_identity}, set())
        self.assertEqual(tree, set())
        self.assertEqual(rss, 0)

        child_reused = dict(snapshot)
        child_reused.pop(root_identity[0])
        child_reused[child_identity[0]] = (os.getpid(), 41002, 8192, 999)
        with patch.object(runner, "_proc_snapshot", return_value=child_reused):
            tree, rss = runner._live_process_tree(root_identity, {child_identity}, set())
        self.assertEqual(tree, set())
        self.assertEqual(rss, 0)

        with (
            patch.object(runner, "_read_process_identity", return_value=(child_identity[0], 999)),
            patch.object(os, "pidfd_open", return_value=91) as pidfd_open,
            patch.object(signal, "pidfd_send_signal") as send_signal,
            patch.object(os, "close"),
            patch.object(os, "waitpid") as waitpid,
        ):
            runner._signal_process_identities({child_identity}, signal.SIGTERM)
            runner._reap_process_identities({child_identity}, root_identity, set())
        pidfd_open.assert_called_once_with(child_identity[0])
        send_signal.assert_not_called()
        waitpid.assert_not_called()

    def test_timeout_does_not_signal_unrelated_child_adopted_after_baseline(self) -> None:
        """A post-baseline subreaper adoption is not proof of runner ownership."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-adopted-unrelated-") as directory:
            cwd = Path(directory)
            root_ready = cwd / "root-ready"
            child_marker = cwd / "unrelated-child.pid"
            child_code = (
                "import os,signal,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(child_marker)!r}).write_text(str(os.getpid()), encoding='ascii'); "
                "time.sleep(10)"
            )
            parent_code = (
                "import os,sys; "
                f"os.posix_spawn(sys.executable, [sys.executable, '-c', {child_code!r}], os.environ.copy())"
            )
            launch_done = threading.Event()

            def launch_unrelated_parent() -> None:
                deadline = time.monotonic() + 5
                while not root_ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                try:
                    subprocess.run([sys.executable, "-c", parent_code], check=True, timeout=2)
                finally:
                    launch_done.set()

            root_code = (
                "import time; from pathlib import Path; "
                f"Path({str(root_ready)!r}).write_text('ready', encoding='ascii'); "
                "time.sleep(30)"
            )
            launcher = threading.Thread(target=launch_unrelated_parent, name="h3-unrelated-adoption")
            launcher.start()
            child_pid: int | None = None
            try:
                result = runner.run_argv(
                    [sys.executable, "-c", root_code],
                    cwd=cwd,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=0.4,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                )
                self.assertEqual(result[0], 124)
                self.assertTrue(launch_done.wait(timeout=2))
                self.assertTrue(child_marker.is_file(), "unrelated child was not created after the runner baseline")
                child_pid = int(child_marker.read_text(encoding="ascii"))
                os.kill(child_pid, 0)
                self.assertTrue(Path(f"/proc/{child_pid}").exists())
            finally:
                launcher.join(timeout=2)
                if child_pid is None and child_marker.exists():
                    child_pid = int(child_marker.read_text(encoding="ascii"))
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(child_pid, 0)
                    except ChildProcessError:
                        pass

    def test_natural_helper_honors_hard_deadline_and_releases_fds(self) -> None:
        """Natural drain may finish a helper, but cannot extend argv timeout."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-natural-hard-deadline-") as directory:
            cwd = Path(directory)
            marker = cwd / "child.pid"
            fd_before = len(os.listdir("/proc/self/fd"))
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            root_code = (
                "import os,subprocess,sys; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(child.pid))"
            )
            natural_started = threading.Event()
            natural = runner._natural_drain_and_reap

            def observe_natural(*args: object, **kwargs: object) -> tuple[bool, int, bool, bool]:
                natural_started.set()
                return natural(*args, **kwargs)

            with (
                patch.object(runner, "_NATURAL_DRAIN_SECONDS", 10.0),
                patch.object(runner, "_natural_drain_and_reap", side_effect=observe_natural),
            ):
                result = runner.run_argv(
                    [sys.executable, "-c", root_code],
                    cwd=cwd,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=2.0,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                )
            self.assertTrue(natural_started.is_set())
            self.assertEqual(result[0], 124)
            self.assertTrue(result[4])
            child_pid = int(marker.read_text(encoding="ascii"))
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
            self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)

    def test_natural_drain_refuses_delayed_snapshot_near_hard_deadline(self) -> None:
        """A non-cancellable /proc walk is never started with insufficient time."""

        selector = selectors.DefaultSelector()
        delayed_calls = 0
        clock = iter((100.0, 100.09))

        def delayed_observation(*_args: object, **_kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
            nonlocal delayed_calls
            delayed_calls += 1
            time.sleep(0.25)
            return set(), 0

        try:
            with (
                patch.object(runner.time, "monotonic", side_effect=lambda: next(clock)),
                patch.object(runner, "_NATURAL_DRAIN_SNAPSHOT_GUARD_SECONDS", 0.02),
                patch.object(runner, "_live_process_tree", side_effect=delayed_observation),
            ):
                complete, _bytes, overflow, timed_out = runner._natural_drain_and_reap(
                    None,  # type: ignore[arg-type]  # not touched before the guard returns
                    selector,
                    {"stdout": bytearray(), "stderr": bytearray()},
                    1024,
                    0,
                    (41001, 101),
                    {(41001, 101), (41002, 102)},
                    set(),
                    100.10,
                )
            self.assertFalse(complete)
            self.assertFalse(overflow)
            self.assertTrue(timed_out)
            self.assertEqual(delayed_calls, 0)
        finally:
            selector.close()

    def test_natural_drain_timeout_decision_precedes_delayed_cleanup(self) -> None:
        """Timeout is decided near its hard deadline; reaping grace is separate."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-natural-decision-") as directory:
            cwd = Path(directory)
            marker = cwd / "child.pid"
            fd_before = len(os.listdir("/proc/self/fd"))
            child_pid: int | None = None
            natural_active = threading.Event()
            delayed_observations = 0
            decision_times: list[float] = []
            hard_deadlines: list[float] = []
            real_natural = runner._natural_drain_and_reap
            real_live = runner._live_process_tree
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            root_code = (
                "import os,subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(child.pid)); "
                "time.sleep(0.20)"
            )

            def observe_natural(*args: object, **kwargs: object) -> tuple[bool, int, bool, bool]:
                hard_deadlines.append(float(args[-1]))
                natural_active.set()
                try:
                    result = real_natural(*args, **kwargs)
                    decision_times.append(time.monotonic())
                    return result
                finally:
                    natural_active.clear()

            def delayed_live(*args: object, **kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
                nonlocal delayed_observations
                if natural_active.is_set():
                    delayed_observations += 1
                    time.sleep(0.25)
                return real_live(*args, **kwargs)

            try:
                with (
                    patch.object(runner, "_NATURAL_DRAIN_SNAPSHOT_GUARD_SECONDS", 0.23),
                    patch.object(runner, "_natural_drain_and_reap", side_effect=observe_natural),
                    patch.object(runner, "_live_process_tree", side_effect=delayed_live),
                ):
                    result = runner.run_argv(
                        [sys.executable, "-c", root_code],
                        cwd=cwd,
                        env={"PATH": "/usr/bin:/bin"},
                        timeout=0.40,
                        rss_limit=256 * 1024 * 1024,
                        output_limit=1024,
                    )
                child_pid = int(marker.read_text(encoding="ascii"))
                self.assertEqual(result[0], 124)
                self.assertTrue(result[4])
                self.assertEqual(delayed_observations, 0)
                self.assertEqual(len(decision_times), 1)
                self.assertEqual(len(hard_deadlines), 1)
                self.assertGreaterEqual(decision_times[0], hard_deadlines[0] - 0.25)
                self.assertLessEqual(decision_times[0], hard_deadlines[0] + 0.05)
                self.assertGreater(result[3], 0.40)  # SIGTERM/SIGKILL cleanup grace is not decision delay.
                self.assertLess(result[3], 2.0)
                self.assertFalse(Path(f"/proc/{child_pid}").exists())
                self.assertEqual(len(os.listdir("/proc/self/fd")), fd_before)
            finally:
                if child_pid is None and marker.exists():
                    child_pid = int(marker.read_text(encoding="ascii"))
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(child_pid, 0)
                    except ChildProcessError:
                        pass

        # A slow full-tree scan may finish after the cleanup deadline, but it
        # cannot trigger another scan or extend the 50 ms grace without bound.
        with tempfile.TemporaryDirectory(prefix="sllm-h3-cleanup-deadline-") as directory:
            cwd = Path(directory)
            marker = cwd / "root-child.pid"
            root_pid: int | None = None
            child_pid: int | None = None
            cleanup_active = False
            cleanup_started: float | None = None
            cleanup_scan_starts: list[float] = []
            cleanup_failed_values: list[bool] = []
            real_live = runner._live_process_tree
            real_terminate = runner._terminate_and_reap
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            root_code = (
                "import os,subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                "time.sleep(30)"
            )

            def delayed_cleanup_live(*args: object, **kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
                if cleanup_active:
                    cleanup_scan_starts.append(time.monotonic())
                    time.sleep(0.25)
                return real_live(*args, **kwargs)

            def observe_terminate(*args: object, **kwargs: object) -> tuple[int, int, bool]:
                nonlocal cleanup_active, cleanup_started
                cleanup_started = time.monotonic()
                cleanup_active = True
                try:
                    result = real_terminate(*args, **kwargs)
                    cleanup_failed_values.append(result[2])
                    return result
                finally:
                    cleanup_active = False

            def assert_absent(pid: int) -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                    except FileNotFoundError:
                        return
                    right_paren = stat_text.rfind(")")
                    self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                    time.sleep(0.01)
                self.fail(f"PID {pid} survived deadline-limited cleanup")

            try:
                with (
                    patch.object(runner, "_TERMINATE_AND_REAP_SECONDS", 0.05),
                    patch.object(runner, "_live_process_tree", side_effect=delayed_cleanup_live),
                    patch.object(runner, "_terminate_and_reap", side_effect=observe_terminate),
                ):
                    result = runner.run_argv(
                        [sys.executable, "-c", root_code],
                        cwd=cwd,
                        env={"PATH": "/usr/bin:/bin"},
                        timeout=0.20,
                        rss_limit=256 * 1024 * 1024,
                        output_limit=1024,
                    )
                root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                self.assertEqual(result[0], 124)
                self.assertTrue(result[4])
                self.assertEqual(cleanup_failed_values, [True])
                self.assertIsNotNone(cleanup_started)
                assert cleanup_started is not None
                self.assertEqual(len(cleanup_scan_starts), 1)
                self.assertLess(cleanup_scan_starts[0], cleanup_started + 0.05)
                self.assertLess(time.monotonic() - cleanup_started, 0.80)
                assert_absent(root_pid)
                assert_absent(child_pid)
            finally:
                for pid in (root_pid, child_pid):
                    if pid is None:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass

    def test_normal_monitoring_checks_deadline_before_delayed_full_scan(self) -> None:
        """A delayed scan may overrun, but no normal scan starts after timeout."""

        for timeout in (0.1, 0.3, 0.5):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory(prefix="sllm-h3-normal-deadline-") as directory:
                cwd = Path(directory)
                scan_starts: list[float] = []
                cleanup_started = False
                real_live = runner._live_process_tree
                real_terminate = runner._terminate_and_reap
                real_run = runner._run_argv
                runner_started: list[float] = []
                root_code = "import time; time.sleep(30)"

                def delayed_live(*args: object, **kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
                    if not cleanup_started:
                        scan_starts.append(time.monotonic())
                        time.sleep(0.25)
                    return real_live(*args, **kwargs)

                def observe_terminate(*args: object, **kwargs: object) -> tuple[int, int, bool]:
                    nonlocal cleanup_started
                    cleanup_started = True
                    return real_terminate(*args, **kwargs)

                def observe_run(*args: object, **kwargs: object) -> tuple[int, bytes, bytes, float, bool, int]:
                    runner_started.append(time.monotonic())
                    return real_run(*args, **kwargs)

                with (
                    patch.object(runner, "_TERMINATE_AND_REAP_SECONDS", 0.20),
                    patch.object(runner, "_live_process_tree", side_effect=delayed_live),
                    patch.object(runner, "_terminate_and_reap", side_effect=observe_terminate),
                    patch.object(runner, "_run_argv", side_effect=observe_run),
                ):
                    result = runner.run_argv(
                        [sys.executable, "-c", root_code],
                        cwd=cwd,
                        env={"PATH": "/usr/bin:/bin"},
                        timeout=timeout,
                        rss_limit=256 * 1024 * 1024,
                        output_limit=1024,
                    )
                self.assertEqual(result[0], 124)
                self.assertTrue(result[4])
                self.assertTrue(scan_starts)
                self.assertEqual(len(runner_started), 1)
                self.assertTrue(all(start < runner_started[0] + timeout for start in scan_starts))
                self.assertLess(result[3], timeout + 0.25 + 0.80)

    def test_fake_argv_does_not_touch_preexisting_unrelated_process(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-baseline-") as directory:
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            try:
                result = runner.run_argv(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=Path(directory),
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=0.05,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                )
                self.assertEqual(result[0], 124)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=2)

        # A cleanup exception must not mask an interrupt.  Both cleanup
        # layers still run their real root-safe work before reporting their
        # injected internal failures as notes on the original exception.
        with tempfile.TemporaryDirectory(prefix="sllm-h3-base-cleanup-failures-") as directory:
            cwd = Path(directory)
            marker = cwd / "root-child.pid"
            root_pid: int | None = None
            child_pid: int | None = None
            injected = False
            real_live = runner._live_process_tree
            real_terminate = runner._terminate_and_reap
            real_emergency = runner._emergency_cleanup_after_observation_failure
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            root_code = (
                "import os,subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                "time.sleep(30)"
            )

            def interrupt_after_bind(*args: object, **kwargs: object) -> tuple[set[runner.ProcessIdentity], int]:
                nonlocal injected, root_pid, child_pid
                if not injected:
                    injected = True
                    deadline = time.monotonic() + 3
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                    self.assertTrue(marker.exists(), "cleanup-failure root did not publish identities")
                    root_pid, child_pid = (int(item) for item in marker.read_text(encoding="ascii").split(":", 1))
                    raise KeyboardInterrupt("cleanup chain")
                return real_live(*args, **kwargs)

            def terminate_then_fail(*args: object, **kwargs: object) -> tuple[int, int, bool]:
                real_terminate(*args, **kwargs)
                raise runner.RuntimeContractError("injected termination cleanup failure")

            def emergency_then_fail(*args: object, **kwargs: object) -> bool:
                real_emergency(*args, **kwargs)
                raise runner.RuntimeContractError("injected emergency cleanup failure")

            def assert_absent(pid: int) -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                    except FileNotFoundError:
                        return
                    right_paren = stat_text.rfind(")")
                    self.assertNotEqual(stat_text[right_paren + 2 :].split()[0], "Z", f"PID {pid} remained a zombie")
                    time.sleep(0.01)
                self.fail(f"PID {pid} survived layered cleanup failure")

            try:
                with (
                    patch.object(runner, "_live_process_tree", side_effect=interrupt_after_bind),
                    patch.object(runner, "_terminate_and_reap", side_effect=terminate_then_fail),
                    patch.object(runner, "_emergency_cleanup_after_observation_failure", side_effect=emergency_then_fail),
                    self.assertRaises(KeyboardInterrupt) as raised,
                ):
                    runner.run_argv(
                        [sys.executable, "-c", root_code],
                        cwd=cwd,
                        env={"PATH": "/usr/bin:/bin"},
                        timeout=5,
                        rss_limit=256 * 1024 * 1024,
                        output_limit=1024,
                    )
                self.assertEqual(raised.exception.args, ("cleanup chain",))
                notes = "\n".join(getattr(raised.exception, "__notes__", []))
                self.assertIn("injected termination cleanup failure", notes)
                self.assertIn("injected emergency cleanup failure", notes)
                assert root_pid is not None and child_pid is not None
                assert_absent(root_pid)
                assert_absent(child_pid)
            finally:
                for pid in (root_pid, child_pid):
                    if pid is None:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass

    def test_fake_argv_fails_closed_when_proc_observation_is_unavailable(self) -> None:
        with patch.object(runner, "_proc_snapshot", side_effect=runner.RuntimeContractError("/proc unavailable")):
            with self.assertRaises(runner.RuntimeContractError):
                runner.run_argv(
                    [sys.executable, "-c", "pass"],
                    cwd=Path(tempfile.gettempdir()),
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=1,
                    rss_limit=256 * 1024 * 1024,
                    output_limit=1024,
                )

    def test_observation_failure_immediately_after_spawn_kills_audited_root_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-observe-immediate-") as directory:
            cwd = Path(directory)
            marker = cwd / "root.pid"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            calls = 0
            emergency_calls = 0
            original_snapshot = runner._proc_snapshot
            original_emergency = runner._emergency_cleanup_after_observation_failure
            subreaper_before = runner._child_subreaper_enabled()

            def fail_after_spawn() -> dict[int, tuple[int, int, int, int]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_snapshot()
                deadline = time.monotonic() + 2
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                raise runner.RuntimeContractError("/proc failed immediately after spawn")

            def observe_emergency(*args: object, **kwargs: object) -> bool:
                nonlocal emergency_calls
                emergency_calls += 1
                return original_emergency(*args, **kwargs)

            try:
                code = (
                    "import os,time; "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid())); "
                    "time.sleep(30)"
                )
                with (
                    patch.object(runner, "_proc_snapshot", side_effect=fail_after_spawn),
                    patch.object(runner, "_terminate_and_reap", side_effect=AssertionError("RuntimeContractError must not enter normal cleanup")),
                    patch.object(runner, "_emergency_cleanup_after_observation_failure", side_effect=observe_emergency),
                ):
                    with self.assertRaises(runner.RuntimeContractError):
                        runner.run_argv(
                            [sys.executable, "-c", code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                self.assertGreaterEqual(calls, 2)
                self.assertEqual(emergency_calls, 1, "RuntimeContractError cleanup ran more than once")
                audited_pid = int(marker.read_text(encoding="ascii"))
                for _ in range(40):
                    if not Path(f"/proc/{audited_pid}").exists():
                        break
                    time.sleep(0.025)
                else:
                    self.fail(f"audited root process {audited_pid} survived emergency cleanup")
                self.assertIsNone(unrelated.poll())
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=2)

    def test_pre_observation_failure_skips_baseline_zombie_and_reaps_audited_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-baseline-zombie-") as directory:
            cwd = Path(directory)
            root_marker = cwd / "root.pid"
            baseline = subprocess.Popen([sys.executable, "-c", "pass"])
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            baseline_waited = False
            calls = 0
            original_snapshot = runner._proc_snapshot
            subreaper_before = runner._child_subreaper_enabled()

            def state(pid: int) -> str | None:
                try:
                    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                except FileNotFoundError:
                    return None
                right_paren = stat_text.rfind(")")
                return stat_text[right_paren + 2 :].split()[0]

            def fail_before_first_post_spawn_snapshot() -> dict[int, tuple[int, int, int, int]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_snapshot()
                deadline = time.monotonic() + 2
                while not root_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                raise runner.RuntimeContractError("/proc failed before first post-spawn snapshot")

            try:
                for _ in range(80):
                    if state(baseline.pid) == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail("baseline child did not become a zombie")
                child_code = "import time; time.sleep(30)"
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(root_marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )
                with patch.object(runner, "_proc_snapshot", side_effect=fail_before_first_post_spawn_snapshot):
                    with self.assertRaises(runner.RuntimeContractError):
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                self.assertGreaterEqual(calls, 2)
                root_pid, child_pid = (int(value) for value in root_marker.read_text(encoding="ascii").split(":", 1))
                for audited_pid in (root_pid, child_pid):
                    for _ in range(80):
                        if state(audited_pid) is None:
                            break
                        time.sleep(0.025)
                    self.assertIsNone(state(audited_pid), f"audited PID {audited_pid} remained present")
                self.assertIsNone(unrelated.poll())
                self.assertEqual(state(baseline.pid), "Z")
                baseline_rc = baseline.wait(timeout=2)
                baseline_waited = True
                self.assertEqual(baseline_rc, 0)
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                if not baseline_waited:
                    try:
                        os.kill(baseline.pid, 9)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(baseline.pid, 0)
                    except ChildProcessError:
                        pass
                unrelated.terminate()
                unrelated.wait(timeout=2)

    def test_observation_failure_before_first_post_spawn_snapshot_contains_setsid_descendant(self) -> None:
        """A pre-observation setsid attempt is denied by inherited containment."""

        with tempfile.TemporaryDirectory(prefix="sllm-h3-observe-setsid-") as directory:
            cwd = Path(directory)
            root_marker = cwd / "root.pid"
            outcome = cwd / "setsid.outcome"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            calls = 0
            original_snapshot = runner._proc_snapshot
            subreaper_before = runner._child_subreaper_enabled()

            def fail_before_first_post_spawn_snapshot() -> dict[int, tuple[int, int, int, int]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_snapshot()
                deadline = time.monotonic() + 2
                while not outcome.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                raise runner.RuntimeContractError("/proc failed before first post-spawn snapshot")

            try:
                child_code = (
                    "import os,time\n"
                    "marker = " + repr(str(outcome)) + "\n"
                    "try:\n os.setsid()\n"
                    "except PermissionError:\n open(marker, 'w').write(str(os.getpid()) + ':denied')\n"
                    "else:\n open(marker, 'w').write(str(os.getpid()) + ':escaped')\n"
                    "time.sleep(30)"
                )
                root_code = (
                    "import os,subprocess,sys,time; "
                    f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    f"open({str(root_marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )
                with patch.object(runner, "_proc_snapshot", side_effect=fail_before_first_post_spawn_snapshot):
                    with self.assertRaises(runner.RuntimeContractError):
                        runner.run_argv(
                            [sys.executable, "-c", root_code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=5,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                self.assertGreaterEqual(calls, 2)
                self.assertTrue(outcome.read_text(encoding="ascii").endswith(":denied"))
                root_pid, child_pid = (int(value) for value in root_marker.read_text(encoding="ascii").split(":", 1))
                for audited_pid in (root_pid, child_pid):
                    for _ in range(40):
                        try:
                            state_line = Path(f"/proc/{audited_pid}/stat").read_text(encoding="ascii")
                            right_paren = state_line.rfind(")")
                            state = state_line[right_paren + 2 :].split()[0]
                            if state == "Z":
                                self.fail(f"contained process {audited_pid} was left as a zombie")
                        except (FileNotFoundError, ProcessLookupError):
                            break
                        time.sleep(0.025)
                    else:
                        self.fail(f"contained process {audited_pid} survived observation failure")
                self.assertIsNone(unrelated.poll())
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=2)

    def test_cleanup_observation_failure_kills_root_and_known_descendants(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-observe-cleanup-") as directory:
            cwd = Path(directory)
            marker = cwd / "root.pid"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            calls = 0
            original_snapshot = runner._proc_snapshot
            subreaper_before = runner._child_subreaper_enabled()

            def fail_during_cleanup() -> dict[int, tuple[int, int, int, int]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_snapshot()
                if calls == 2:
                    deadline = time.monotonic() + 2
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                    time.sleep(0.05)
                    return original_snapshot()
                raise runner.RuntimeContractError("/proc failed during cleanup")

            try:
                code = (
                    "import os,subprocess,sys,time; "
                    "child_code = 'import os,time; os.setsid(); time.sleep(30)'; "
                    "child = subprocess.Popen([sys.executable, '-c', child_code]); "
                    f"open({str(marker)!r}, 'w').write(str(os.getpid()) + ':' + str(child.pid)); "
                    "time.sleep(30)"
                )
                with patch.object(runner, "_proc_snapshot", side_effect=fail_during_cleanup):
                    with self.assertRaises(runner.RuntimeContractError):
                        runner.run_argv(
                            [sys.executable, "-c", code],
                            cwd=cwd,
                            env={"PATH": "/usr/bin:/bin"},
                            timeout=0.01,
                            rss_limit=256 * 1024 * 1024,
                            output_limit=1024,
                        )
                self.assertGreaterEqual(calls, 3)
                root_pid, child_pid = (int(value) for value in marker.read_text(encoding="ascii").split(":", 1))
                for audited_pid in (root_pid, child_pid):
                    for _ in range(40):
                        try:
                            state_line = Path(f"/proc/{audited_pid}/stat").read_text(encoding="ascii")
                            right_paren = state_line.rfind(")")
                            state = state_line[right_paren + 2 :].split()[0]
                            if state == "Z":
                                self.fail(f"audited process {audited_pid} was left as a zombie")
                        except (FileNotFoundError, ProcessLookupError):
                            break
                        time.sleep(0.025)
                    else:
                        self.fail(f"audited process {audited_pid} survived cleanup emergency path")
                self.assertIsNone(unrelated.poll())
                self.assertEqual(runner._child_subreaper_enabled(), subreaper_before)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=2)

    def test_process_observation_scope_is_reentrant_and_restores_subreaper_state(self) -> None:
        before = runner._child_subreaper_enabled()
        with runner._process_observation_scope() as outer_baseline:
            self.assertTrue(any(identity[0] == os.getpid() for identity in outer_baseline))
            with runner._process_observation_scope() as inner_baseline:
                self.assertTrue(any(identity[0] == os.getpid() for identity in inner_baseline))
                self.assertTrue(runner._child_subreaper_enabled())
        self.assertEqual(runner._child_subreaper_enabled(), before)

    def test_device_inspection_rejects_wrong_target_codegen_and_public_runtime_symbols(self) -> None:
        row = {"target": "gfx1030", "resource": {"max_rss_bytes": 256 * 1024 * 1024, "max_output_bytes": 1024}}
        with tempfile.TemporaryDirectory(prefix="sllm-h3-device-") as directory:
            path = Path(directory) / "device.elf"
            path.write_bytes(b"fake-device")
            with patch.object(runner, "readobj", return_value=fake_device_readobj()):
                report = runner.inspect_device(path, Path("/fake/llvm-readobj"), row)
            self.assertEqual(report["target"], "gfx1030")
            for fake in (
                fake_device_readobj(target="gfx1201"),
                fake_device_readobj(flags="0x0000004e"),
                fake_device_readobj(wave=64),
                fake_device_readobj(public_symbol=True),
                fake_device_readobj(duplicate_probe=True),
                fake_device_readobj(duplicate_dynamic=True),
                fake_device_readobj().replace(
                    "Name: _DYNAMIC\n    Binding: Local (0x0)\n    Type: None (0x0)\n    Other: 2\n    Section: .dynamic",
                    "Name: _DYNAMIC\n    Binding: Global (0x1)\n    Type: None (0x0)\n    Other: 2\n    Section: .dynamic",
                    1,
                ),
                fake_device_readobj(extra_symbol="injected_kernel"),
                fake_device_readobj(extra_symbol="bad.symbol"),
                fake_device_readobj(extra_symbol=".local_label"),
                fake_device_readobj(extra_symbol="name$with$chars"),
                fake_device_readobj(extra_symbol="9starts_with_digit"),
                fake_device_readobj(target_note="gfx1030:xnack-"),
                fake_device_readobj(extra_target_note="gfx1201"),
                fake_device_readobj(omit_target=True),
                fake_device_readobj(omit_flags=True),
                fake_device_readobj(flags="0x00000336"),
                fake_device_readobj(feature_lines="  xnack: on"),
                fake_device_readobj(feature_lines="  generic_processor_version: 1"),
            ):
                with self.subTest(fake=fake.splitlines()[0:4]):
                    with patch.object(runner, "readobj", return_value=fake), self.assertRaises(runner.RuntimeContractError):
                        runner.inspect_device(path, Path("/fake/llvm-readobj"), row)

    def test_device_cuid_accepts_only_exact_15_or_16_lowercase_hex_roles(self) -> None:
        for cuid_name in (
            "__hip_cuid_f41a4806ab0eb4b",
            "__hip_cuid_0123456789abcdef",
        ):
            with self.subTest(cuid_name=cuid_name):
                self.assertEqual(
                    runner._require_device_symbols(fake_device_readobj(cuid_name=cuid_name)),
                    ["sllm_hip_compile_probe"],
                )

        invalid_records = (
            (fake_device_readobj(cuid_name="__hip_cuid_"), "empty CUID suffix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0"), "one-digit CUID suffix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0123456789abcd"), "14-digit CUID suffix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0123456789abcdeg"), "nonhex CUID suffix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0123456789abcdeF"), "uppercase CUID suffix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0123456789abcdef0"), "17-digit CUID suffix"),
            (fake_device_readobj(cuid_name="prefix__hip_cuid_0123456789abcdef"), "CUID prefix"),
            (fake_device_readobj(cuid_name="__hip_cuid_0123456789abcdef_suffix"), "CUID suffix"),
            (fake_device_readobj(duplicate_cuid=True), "duplicate CUID"),
            (fake_device_readobj(cuid_binding="Local (0x0)"), "CUID binding"),
            (fake_device_readobj(cuid_type="Function (0x2)"), "CUID type"),
            (fake_device_readobj(cuid_section=".data"), "CUID section"),
            (fake_device_readobj(cuid_other="2"), "CUID visibility"),
            (fake_device_readobj(extra_symbol="unexpected_device_symbol"), "unexpected defined symbol"),
            (fake_device_readobj(extra_undefined_symbol="unexpected_undefined_symbol"), "unexpected undefined symbol"),
        )
        for output, label in invalid_records:
            with self.subTest(label=label), self.assertRaises(runner.RuntimeContractError):
                runner._require_device_symbols(output)

    def test_real_rocm_symbol_visibility_forms_and_generated_roles_are_exact(self) -> None:
        row = {"target": "gfx1030", "resource": {"max_rss_bytes": 256 * 1024 * 1024, "max_output_bytes": 1024}}
        with tempfile.TemporaryDirectory(prefix="sllm-h3-real-symbols-") as directory:
            path = Path(directory) / "device.elf"
            path.write_bytes(b"fake-device")
            actual_like = fake_device_readobj().replace(
                "    Name: sllm_hip_compile_probe\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: .text",
                "    Name: sllm_hip_compile_probe\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other [ (0x3)\n      STV_PROTECTED (0x3)\n    ]\n    Section: .text",
                1,
            )
            with patch.object(runner, "readobj", return_value=actual_like):
                report = runner.inspect_device(path, Path("/fake/llvm-readobj"), row)
            self.assertEqual(report["symbols"], [{"name": "sllm_hip_compile_probe", "defined": True}])
            for malformed in (
                actual_like.replace("STV_PROTECTED (0x3)", "STV_HIDDEN (0x2)", 1),
                actual_like.replace("STV_PROTECTED (0x3)", "STV_PROTECTED (0x2)", 1),
                actual_like.replace("    ]\n    Section: .text", "    ]\n    Visibility: Hidden (0x2)\n    Section: .text", 1),
                actual_like.replace("sllm_hip_compile_probe.kd", "sllm_arbitrary.kd", 1),
                actual_like.replace("__hip_cuid_0123456789abcdef", "__hip_cuid_not-a-cuid", 1),
            ):
                with self.subTest(malformed=malformed.splitlines()[0:4]):
                    with patch.object(runner, "readobj", return_value=malformed), self.assertRaises(runner.RuntimeContractError):
                        runner.inspect_device(path, Path("/fake/llvm-readobj"), row)

    def test_host_inspection_requires_public_symbols_and_rejects_stub_names(self) -> None:
        self.assertIn("hipEventElapsedTime", runner.EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS)
        row = {"target": "gfx1030", "resource": {"max_rss_bytes": 256 * 1024 * 1024, "max_output_bytes": 1024}}
        symbol_text = "\n".join(
            f"  Symbol {{\n    Name: {name}\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: .text\n  }}" for name in sorted(runner.PUBLIC_SYMBOLS)
        )
        def host_symbol(name: str, symbol_type: str, section: str) -> str:
            return f"  Symbol {{\n    Name: {name}\n    Binding: Global (0x1)\n    Type: {symbol_type}\n    Other: 0\n    Section: {section}\n  }}"

        undefined_host_hip_symbols = "\n".join(
            host_symbol(name, "None (0x0)", "Undefined (0x0)")
            for name in runner.EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS
        )
        host_probe_symbol = "  Symbol {\n    Name: sllm_hip_compile_probe\n    Binding: Global (0x1)\n    Type: Object (0x1)\n    Other: 0\n    Section: .data.rel.ro\n  }"
        host_kernel_symbol = "  Symbol {\n    Name: sllm_rmsnorm_baseline_wave32_v1\n    Binding: Global (0x1)\n    Type: Object (0x1)\n    Other: 0\n    Section: .data.rel.ro\n  }"
        host_text = f"""FileHeaders [
  Class: ELF64
  Arch: x86_64
]
Sections [
  Section {{
    Name: .text
    Size: 0x10
  }}
  Section {{
    Name: .hip_fatbin
    Size: 0x10
  }}
]
{symbol_text}
{undefined_host_hip_symbols}
{host_probe_symbol}
{host_kernel_symbol}
"""
        with tempfile.TemporaryDirectory(prefix="sllm-h3-host-") as directory:
            path = Path(directory) / "host.elf"
            path.write_bytes(b"fake-host")
            with patch.object(runner, "readobj", return_value=host_text):
                report = runner.inspect_host(path, Path("/fake/llvm-readobj"), row, [runner.BUNDLE_IDS["gfx1030"], runner.HOST_BUNDLE_ID])
            self.assertEqual(report["machine"], "X86_64")
            self.assertEqual(report["bundles"], [runner.BUNDLE_IDS["gfx1030"], runner.HOST_BUNDLE_ID])
            expected_bundles = [runner.BUNDLE_IDS["gfx1030"], runner.HOST_BUNDLE_ID]
            compiler_stub_text = "\n".join(
                f"  Symbol {{\n    Name: __device_stub__{name}\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: .text\n  }}"
                for name in ("sllm_hip_compile_probe", "sllm_rmsnorm_baseline_wave32_v1")
            )
            with patch.object(runner, "readobj", return_value=host_text + compiler_stub_text):
                compiler_stub_report = runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)
            self.assertEqual(compiler_stub_report["stub_symbols"], [])
            hip_mutations = (
                (host_text.replace(host_symbol("hipMalloc", "None (0x0)", "Undefined (0x0)"), "", 1), "missing HIP runtime symbol"),
                (host_text + host_symbol("hipUnexpectedRuntimeEntry", "None (0x0)", "Undefined (0x0)"), "extra HIP runtime symbol"),
                (host_text + host_symbol("hipMalloc", "None (0x0)", "Undefined (0x0)"), "duplicate HIP runtime symbol"),
                (host_text.replace(host_symbol("hipMalloc", "None (0x0)", "Undefined (0x0)"), host_symbol("hipMalloc", "Function (0x2)", ".text"), 1), "defined instead of undefined"),
                (host_text + host_symbol("hipMallocExtra", "None (0x0)", "Undefined (0x0)"), "near-prefix HIP runtime symbol"),
                (host_text + host_symbol("hipMalloc<T>", "None (0x0)", "Undefined (0x0)"), "template-like HIP runtime symbol"),
                (host_text + host_symbol("__hipRegisterFunctionSuffix", "None (0x0)", "Undefined (0x0)"), "near-prefix compiler runtime symbol"),
            )
            for mutated, label in hip_mutations:
                with self.subTest(label=label), patch.object(runner, "readobj", return_value=mutated), self.assertRaises(runner.RuntimeContractError):
                    runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)
            with patch.object(
                runner,
                "readobj",
                return_value=host_text + "\n" + host_symbol("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", "None (0x0)", "Undefined (0x0)") + "\n" + host_symbol("memcpy", "None (0x0)", "Undefined (0x0)"),
            ):
                unrelated_undefined_report = runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)
            self.assertEqual(unrelated_undefined_report["public_symbols"], [{"name": name, "defined": True} for name in sorted(runner.PUBLIC_SYMBOLS)])
            for mutated in (
                host_text.replace("sllm_context_create", "sllm_context_missing", 1),
                host_text.replace("Name: sllm_context_create", "Name: sllm_public_runtime_stub", 1),
                host_text.replace("Name: sllm_rmsnorm_execute", "Name: sllm_rmsnorm_execute_missing", 1),
                host_text + host_text[host_text.index("  Symbol {\n    Name: sllm_rmsnorm_execute") : host_text.index("  Symbol {\n    Name: sllm_rmsnorm_execute") + host_text[host_text.index("  Symbol {\n    Name: sllm_rmsnorm_execute") :].index("  }") + 4],
                host_text.replace(
                    "  Symbol {\n    Name: sllm_rmsnorm_execute\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: .text\n  }",
                    "  Symbol {\n    Name: sllm_rmsnorm_execute\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: Undefined (0x0)\n  }",
                    1,
                ),
                host_text.replace(
                    "  Symbol {\n    Name: sllm_rmsnorm_execute\n    Binding: Global (0x1)\n    Type: Function (0x2)\n    Other: 0\n    Section: .text\n  }",
                    "  Symbol {\n    Name: sllm_rmsnorm_execute\n    Binding: Global (0x1)\n    Type: Object (0x1)\n    Other: 0\n    Section: .text\n  }",
                    1,
                ),
                host_text.replace("Name: sllm_rmsnorm_baseline_wave32_v1", "Name: sllm_wrong_kernel", 1),
                host_text + host_kernel_symbol,
                host_text.replace(
                    "Name: sllm_rmsnorm_baseline_wave32_v1\n    Binding: Global (0x1)\n    Type: Object (0x1)",
                    "Name: sllm_rmsnorm_baseline_wave32_v1\n    Binding: Global (0x1)\n    Type: Function (0x2)",
                    1,
                ),
                host_text.replace(host_kernel_symbol, host_kernel_symbol.replace("Section: .data.rel.ro", "Section: Undefined (0x0)"), 1),
                host_text.replace("Section: .text", "Section: Undefined (0x0)", 1),
                host_text.replace("Type: Function (0x2)", "Type: Object (0x1)", 1),
                host_text.replace("Binding: Global (0x1)", "Binding: Local (0x0)", 1),
                host_text.replace("Other: 0", "Other: 1", 1),
                host_text.replace("    Type: Function (0x2)\n", "", 1),
                host_text + host_text[host_text.index("  Symbol {") : host_text.index("  Symbol {") + host_text[host_text.index("  Symbol {") :].index("  }") + 4],
            ):
                with self.subTest(mutated=mutated[-80:]):
                    with patch.object(runner, "readobj", return_value=mutated), self.assertRaises(runner.RuntimeContractError):
                        runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)
            with patch.object(runner, "readobj", return_value=host_text), self.assertRaises(runner.RuntimeContractError):
                runner.inspect_host(path, Path("/fake/llvm-readobj"), row, [runner.BUNDLE_IDS["gfx1030"], "host-other"])

            multiline_host = host_text.replace(
                "    Other: 0\n    Section: .text",
                "    Other [ (0x0)\n      STV_DEFAULT (0x0)\n    ]\n    Section: .text",
                1,
            )
            with patch.object(runner, "readobj", return_value=multiline_host):
                multiline_report = runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)
            self.assertEqual(multiline_report["public_symbols"], [{"name": name, "defined": True} for name in sorted(runner.PUBLIC_SYMBOLS)])
            for malformed in (
                multiline_host.replace("STV_DEFAULT (0x0)", "STV_HIDDEN (0x2)", 1),
                multiline_host.replace("STV_DEFAULT (0x0)", "STV_DEFAULT (0x1)", 1),
                host_text.replace("Section: .data.rel.ro", "Section: .bss", 1),
                host_text.replace("Type: Object (0x1)", "Type: Function (0x2)", 1),
            ):
                with self.subTest(malformed=malformed[-100:]):
                    with patch.object(runner, "readobj", return_value=malformed), self.assertRaises(runner.RuntimeContractError):
                        runner.inspect_host(path, Path("/fake/llvm-readobj"), row, expected_bundles)

    def test_output_and_source_paths_reject_symlink_components_and_workspace_escapes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-paths-") as directory:
            root = Path(directory) / "repo"
            outside = Path(directory) / "outside"
            (root / "include").mkdir(parents=True)
            outside.mkdir()
            source = root / "include" / "hip.h"
            source.write_text("source", encoding="ascii")
            with self.assertRaises(runner.RuntimeContractError):
                runner._validate_output_directory(root / "output", root)
            symlink_parent = Path(directory) / "parent-link"
            symlink_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(runner.RuntimeContractError):
                runner._validate_output_directory(symlink_parent / "output", root)
            symlink_leaf = Path(directory) / "leaf-link"
            symlink_leaf.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(runner.RuntimeContractError):
                runner._validate_output_directory(symlink_leaf, root)
            escaped_include = root / "src-link"
            escaped_include.symlink_to(outside, target_is_directory=True)
            escaped_source = escaped_include / "hip.h"
            (outside / "hip.h").write_text("outside", encoding="ascii")
            with self.assertRaises(runner.RuntimeContractError):
                runner.require_regular(escaped_source, "escaped source", within=root)

    def test_matrix_rendering_is_exact_two_target_compile_link_and_never_output_execution(self) -> None:
        _toolchain, matrix, rows = runner.validate_matrix(ROOT)
        self.assertEqual(tuple(matrix["targets"]), runner.TARGETS)
        self.assertEqual(tuple(matrix["direct_compile_sources"]["canonical_order"]), runner.EXPECTED_DIRECT_COMPILE_SOURCE_PATHS)
        for target in runner.TARGETS:
            commands = runner.render_commands(rows[f"h3-public-{target}"], ROOT, Path("/tmp/private-build"))
            expected = []
            for template in runner.expected_build_commands():
                expected.append([
                    token.replace("{repo}", str(ROOT)).replace("{build_dir}", "/tmp/private-build").replace("{target}", target)
                    for token in template
                ])
            self.assertEqual(commands, expected)
            self.assertEqual(len(commands), 5)
            self.assertEqual(commands[-1][11:15], [
                "/tmp/private-build/hip-compile-probe-" + target + ".o",
                "/tmp/private-build/public-runtime-" + target + ".o",
                "/tmp/private-build/rmsnorm-kernel-" + target + ".o",
                "/tmp/private-build/rmsnorm-api-" + target + ".o",
            ])
            self.assertNotIn("native/hip/src/rmsnorm_api.cpp", commands[-1])
            self.assertFalse(any("./" in token or "--run" in token for command in commands for token in command))


if __name__ == "__main__":
    unittest.main()
