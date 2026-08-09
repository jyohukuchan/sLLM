from __future__ import annotations

import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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
        success = builder._run_bounded_build(
            [sys.executable, "-c", "import sys; sys.stdout.write('ok')"],
            cwd=ROOT,
            env=environment,
        )
        self.assertEqual((success.returncode, success.stdout, success.stderr), (0, b"ok", b""))

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
                "import pathlib,signal,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
                "time.sleep(60)"
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
            with self.assertRaisesRegex(contracts.ContractError, "output exceeded"):
                builder._run_bounded_build(
                    [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
                    cwd=ROOT,
                    env=environment,
                )

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
