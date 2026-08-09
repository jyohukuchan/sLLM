"""Host-only adversarial tests for the semantic G1 compiler broker."""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
import socket
import signal
import struct
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import build_rmsnorm_g1_runtime as builder  # noqa: E402
import validate_rmsnorm_g1_contracts as contracts  # noqa: E402


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Keep generic exact-action adversarial coverage in the registered G1 suite."""

    del pattern
    tests.addTests(loader.loadTestsFromName("ci.tests.test_rmsnorm_g1_exact_actions"))
    return tests


class SemanticG1BuilderTests(unittest.TestCase):
    def _broker(
        self,
        root: Path,
        compiler_path: str = "/bin/sleep",
        *,
        compiler_environment: dict[str, str] | None = None,
        require_complete_recipe_set: bool = False,
        action_recipes: dict[str, dict[str, object]] | None = None,
    ) -> tuple[contracts.SealedDescriptor, builder.CompilerBroker, Path, dict[str, str]]:
        client = root / "compiler-client.py"
        client.write_text(builder.COMPILER_CLIENT_TEMPLATE, encoding="utf-8")
        client.chmod(0o700)
        helper = root / "compiler-exec-helper"
        completed = subprocess.run(
            ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(helper), "-"],
            input=builder.COMPILER_EXEC_HELPER_SOURCE.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        helper.chmod(0o555)
        compiler = contracts.snapshot_file(Path(compiler_path), None, "host test compiler")
        recipes = action_recipes or {
            f"host-unit-{index}": {"argv": argv, "cwd": str(root), "inputs": [], "implicit": [], "response_files": [], "outputs": []}
            for index, argv in enumerate((["1"], ["2"], ["3"]))
        }
        client_environment = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
        broker = builder.CompilerBroker(
            compiler=compiler, client_path=client, exec_helper=helper, allowed_roots=(root,),
            compiler_environment=compiler_environment or client_environment,
            action_recipes=recipes, require_complete_recipe_set=require_complete_recipe_set,
        )
        broker.start()
        self.addCleanup(broker.abort)
        self.addCleanup(compiler.close)
        environment = {**client_environment, **broker.environment()}
        return compiler, broker, client, environment

    def _marker_compiler(self, root: Path) -> Path:
        """Build a host-only stand-in that leaves proof if it is ever exec'd."""

        compiler = root / "marker-compiler"
        completed = subprocess.run(
            ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(compiler), "-"],
            input=(
                b"#include <cstdlib>\n#include <fstream>\n#include <unistd.h>\n"
                b"int main(){const char* p=std::getenv(\"SLLM_TEST_COMPILER_TRACE\");"
                b"if(p){std::ofstream(p,std::ios::app)<<\"executed\\n\";}sleep(1);return 0;}\n"
            ),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        compiler.chmod(0o555)
        return compiler

    @staticmethod
    @contextmanager
    def _descriptor_at(client: Path, descriptor: int):
        """Temporarily retain client at one exact inherited descriptor number."""

        try:
            previous = os.dup(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
            previous = None
            previous_inheritable = False
        else:
            previous_inheritable = os.get_inheritable(descriptor)
        raw = -1
        try:
            raw = os.open(client, os.O_RDONLY | os.O_CLOEXEC)
            if raw != descriptor:
                os.dup2(raw, descriptor, inheritable=False)
            yield descriptor
        finally:
            if raw >= 0 and raw != descriptor:
                os.close(raw)
            if previous is None:
                os.close(descriptor)
            else:
                os.dup2(previous, descriptor, inheritable=previous_inheritable)
                os.close(previous)

    @staticmethod
    def _run_cmake_semantic_discovery(
        root: Path, trace: Path, descriptor: int, descriptor_text: str
    ) -> subprocess.CompletedProcess[bytes]:
        client_path = f"/proc/self/fd/{descriptor_text}"
        environment = {
            **os.environ,
            "SLLM_HIP_COMPILER_BROKER_CLIENT": client_path,
            "SLLM_HIP_COMPILER_BROKER_CLIENT_FD": descriptor_text,
            "SLLM_HIP_COMPILER_BROKER_SOCKET": str(root / "broker.sock").replace("-", "_"),
            "SLLM_HIP_COMPILER_BROKER_SESSION": "a" * 64,
            "SLLM_HIP_COMPILER_BROKER_TOKEN": "b" * 64,
            "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256": contracts.fd_sha256(descriptor),
            "SLLM_TEST_COMPILER_TRACE": str(trace),
        }
        return subprocess.run(
            [
                "/usr/bin/cmake", "-S", str(ROOT / "native/hip"), "-B", str(root / "cmake-build"),
                "-G", "Unix Makefiles", "-DSLLM_ENABLE_HIP_COMPILE_PROBE=OFF",
                "-DSLLM_ENABLE_HIP_RUNTIME=OFF", "-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=ON",
                "-DSLLM_ENABLE_PUBLIC_RUNTIME_HOST_TEST=OFF", "-DSLLM_SEMANTIC_G1_AUTHORITY=ON",
                "-DROCM_PATH=/opt/rocm", f"-DCMAKE_HIP_COMPILER={client_path}",
                "-DSLLM_HIP_COMPILER_LOGICAL=/opt/rocm/bin/amdclang++",
                "-DCMAKE_HIP_ARCHITECTURES=gfx1030", "-DSLLM_HIP_COMPILE_TARGET=gfx1030",
                "-DSLLM_HIP_CODEGEN_FEATURES=co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0",
            ],
            env=environment,
            pass_fds=(descriptor,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @staticmethod
    def _bind_build(broker: builder.CompilerBroker, process: subprocess.Popen[bytes]) -> builder.runner.LinuxContainment:
        containment = builder.runner.LinuxContainment.begin()
        containment.bind_root(process.pid, process.pid)
        broker.bind_build(process.pid, os.getpgid(process.pid), process=process, containment=containment)
        return containment

    @staticmethod
    def _run_client(
        broker: builder.CompilerBroker,
        client: Path,
        root: Path,
        environment: dict[str, str],
        *arguments: str,
    ) -> tuple[subprocess.Popen[bytes], bytes, bytes]:
        process = subprocess.Popen(
            [broker.client_exec_path, *arguments],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True, pass_fds=broker.child_pass_fds(),
        )
        containment = SemanticG1BuilderTests._bind_build(broker, process)
        stdout, stderr = process.communicate(timeout=15.0)
        if not containment.terminate_and_reap(process):
            raise AssertionError("build containment did not close")
        broker.mark_build_reaped()
        return process, stdout, stderr

    def test_snapshot_rejects_pre_capture_races_and_retains_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-builder-") as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"original bytes")
            expected = contracts.file_identity(path, "test artifact")
            path.write_bytes(b"same-inode replacement")
            with self.assertRaises(contracts.EvidenceError):
                contracts.snapshot_file(path, expected, "same-inode race")
            path.write_bytes(b"original bytes")
            sealed = contracts.snapshot_file(path, expected, "retained artifact")
            self.addCleanup(sealed.close)
            path.write_bytes(b"mutated after snapshot")
            self.assertEqual(contracts.fd_read_all(sealed.fd), b"original bytes")
            self.assertTrue(contracts.descriptor_is_sealed(sealed.fd))

    def test_file_identity_records_reviewed_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-identity-") as temporary:
            root = Path(temporary)
            target = root / "python3.12"
            target.write_bytes(b"reviewed interpreter bytes")
            logical = root / "python3"
            logical.symlink_to(target.name)

            record = contracts.file_identity(logical, "reviewed interpreter")

            self.assertEqual(record["path"], str(logical))
            self.assertEqual(record["resolved_path"], str(target))
            self.assertEqual(record["size_bytes"], len(b"reviewed interpreter bytes"))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(b"reviewed interpreter bytes").hexdigest(),
            )

    def test_reviewed_snapshot_keeps_nested_directories_private_until_final_seal(self) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:") as stream:
            for name in (".agents", ".agents/skills", ".agents/skills/push"):
                member = tarfile.TarInfo(name)
                member.type = tarfile.DIRTYPE
                member.mode = 0o700
                stream.addfile(member)
            payload = b"reviewed skill bytes"
            member = tarfile.TarInfo(".agents/skills/push/SKILL.md")
            member.size = len(payload)
            member.mode = 0o600
            stream.addfile(member, io.BytesIO(payload))

        with tempfile.TemporaryDirectory(prefix="sllm-g1-reviewed-parent-") as temporary:
            parent = Path(temporary)
            candidate = {"reviewed_sha": "a" * 40}
            with mock.patch.object(contracts, "_git_output_bytes", return_value=archive.getvalue()):
                snapshot = builder._materialize_reviewed_snapshot(ROOT, candidate, parent)

            self.assertEqual(
                (snapshot / ".agents/skills/push/SKILL.md").read_bytes(), payload
            )
            for relative in (".agents", ".agents/skills", ".agents/skills/push"):
                self.assertEqual((snapshot / relative).stat().st_mode & 0o777, 0o700)

    def test_brokered_true_round_trip_preserves_compiler_output_and_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-roundtrip-") as temporary:
            root = Path(temporary)
            compiler, broker, client, environment = self._broker(root)
            process, stdout, stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0, repr(broker.failure))
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            transcript = broker.close(build_reaped=True, validate=False)
            self.assertEqual(transcript["request_count"], 1)
            event = transcript["events"][0]
            self.assertEqual(event["compiler"]["exit_code"], 0)
            self.assertEqual(event["compiler"]["stdout_sha256"], hashlib.sha256(stdout).hexdigest())
            self.assertTrue(os.fstat(compiler.fd).st_size > 0)
            identity = event["compiler"]["exec_identity"]
            self.assertTrue(identity["exec_ready"])
            sealed = os.fstat(compiler.fd)
            self.assertEqual((identity["exe_dev"], identity["exe_ino"]), (identity["sealed_dev"], identity["sealed_ino"]))
            self.assertEqual((identity["sealed_dev"], identity["sealed_ino"]), (sealed.st_dev, sealed.st_ino))
            self.assertEqual(identity["pid"], event["compiler"]["pid"])

    def test_manifest_binds_the_final_compiler_environment_not_client_authentication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-environment-") as temporary:
            root = Path(temporary)
            trace = root / "compiler-trace"
            compiler_environment = {
                "PATH": "/usr/bin:/bin", "HOME": "/tmp", "LC_ALL": "C",
                "SLLM_TEST_COMPILER_TRACE": str(trace), "SLLM_HIP_COMPILER_LOGICAL": "/reviewed/amdclang++",
            }
            _compiler, broker, client, environment = self._broker(
                root, str(self._marker_compiler(root)), compiler_environment=compiler_environment,
            )
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0, repr(broker.failure))
            transcript = broker.close(build_reaped=True, validate=False)
            event = transcript["events"][0]
            manifest_environment = dict(event["action_manifest"]["environment"])
            self.assertEqual(manifest_environment, compiler_environment)
            self.assertTrue(trace.is_file(), "the final compiler did not receive the manifest environment")
            self.assertTrue(event["acknowledged"])
            self.assertRegex(event["ack_frame_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn(builder.COMPILER_BROKER_TOKEN_ENV, manifest_environment)
            self.assertEqual(
                event["client_observation"]["environment_sha256"],
                contracts.sha256_json(environment),
            )

    def test_final_compiler_environment_has_no_mutable_config_or_client_credentials(self) -> None:
        environment = builder.compiler_spawn_environment(
            {"PATH": "/usr/bin:/bin", "HOME": "/home/build", "LC_ALL": "C"},
            "/opt/rocm/bin/amdclang++",
        )
        self.assertEqual(environment, {
            "PATH": "/usr/bin:/bin", "LC_ALL": "C",
            "SLLM_HIP_COMPILER_LOGICAL": "/opt/rocm/bin/amdclang++",
        })
        with self.assertRaisesRegex(builder.BuilderError, "mutable input configuration: CLANG_CONFIG_FILE"):
            builder.compiler_spawn_environment(
                {"PATH": "/usr/bin:/bin", "CLANG_CONFIG_FILE": "/tmp/forged.cfg"},
                "/opt/rocm/bin/amdclang++",
            )
        source = (ROOT / "ci/tools/build_rmsnorm_g1_runtime.py").read_text(encoding="utf-8")
        self.assertIn('"--print-resource-dir"', source)
        self.assertIn('"compiler-resource"', source)
        self.assertIn("live_pre_exec_validation", source)

    def test_native_build_cwd_and_output_root_execute_the_reviewed_semantic_route(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-native-route-") as temporary:
            root = Path(temporary)
            project = root / "reviewed-project"
            native_build = root / "native-build"
            project.mkdir()
            native_build.mkdir()
            source = project / "route.hip.cpp"
            source.write_text("// reviewed input\n", encoding="utf-8")
            output = native_build / "route.o"
            marker = root / "output-marker"
            completed = subprocess.run(
                ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(marker), "-"],
                input=(
                    b"#include <cstring>\n#include <fstream>\n#include <unistd.h>\n"
                    b"int main(int argc,char** argv){for(int i=1;i+1<argc;++i)"
                    b"if(std::strcmp(argv[i],\"-o\")==0)std::ofstream(argv[i+1])<<\"native\";"
                    b"sleep(1);return 0;}\n"
                ),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            marker.chmod(0o555)
            client = root / "compiler-client.py"
            client.write_text(builder.COMPILER_CLIENT_TEMPLATE, encoding="utf-8")
            client.chmod(0o700)
            helper = root / "compiler-exec-helper"
            completed = subprocess.run(
                ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(helper), "-"],
                input=builder.COMPILER_EXEC_HELPER_SOURCE.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            helper.chmod(0o555)
            compiler = contracts.snapshot_file(marker, None, "native route compiler")
            self.addCleanup(compiler.close)
            recipe = {
                "argv": ["-c", str(source), "-o", str(output)], "cwd": str(native_build),
                "inputs": [{"role": "translation-unit", "path": str(source)}],
                "implicit": [], "response_files": [], "outputs": [str(output)],
            }
            broker = builder.CompilerBroker(
                compiler=compiler, client_path=client, exec_helper=helper,
                allowed_roots=(project, native_build), output_roots=(native_build,),
                compiler_environment={"PATH": "/usr/bin:/bin"},
                action_recipes={"native-route": recipe}, require_complete_recipe_set=True,
            )
            broker.start()
            self.addCleanup(broker.abort)
            environment = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", **broker.environment()}
            process, _stdout, _stderr = self._run_client(
                broker, client, native_build, environment, *recipe["argv"],
            )
            self.assertEqual(process.returncode, 0, repr(broker.failure))
            transcript = broker.close(build_reaped=True, validate=False)
            self.assertEqual(transcript["events"][0]["action_manifest"]["cwd"]["path"], str(native_build))
            self.assertEqual(
                transcript["events"][0]["compiler"]["invocation"]["materialized_outputs"][0]["path"],
                str(output),
            )
            self.assertEqual(output.read_bytes(), b"native")

    def test_strict_semantic_recipe_set_rejects_missing_or_unacknowledged_actions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-complete-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root, require_complete_recipe_set=True)
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0)
            with self.assertRaisesRegex(builder.BuilderError, "issue and consume every reviewed compiler recipe"):
                broker.close(build_reaped=True, validate=False)

        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-ack-") as temporary:
            root = Path(temporary)
            recipes = {"only": {"argv": ["1"], "cwd": str(root), "inputs": [], "implicit": [], "response_files": [], "outputs": []}}
            _compiler, broker, client, environment = self._broker(
                root, require_complete_recipe_set=True, action_recipes=recipes,
            )
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0)
            broker._pending_deliveries["forged"] = {"delivery": "not-acknowledged"}
            with self.assertRaisesRegex(builder.BuilderError, "clean build lifetime"):
                broker.close(build_reaped=True, validate=False)

    def test_serialized_strict_recipe_set_rejects_missing_or_unconsumed_actions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-serialized-complete-") as temporary:
            root = Path(temporary)
            recipes = {"only": {"argv": ["1"], "cwd": str(root), "inputs": [], "implicit": [], "response_files": [], "outputs": []}}
            _compiler, broker, client, environment = self._broker(
                root, require_complete_recipe_set=True, action_recipes=recipes,
            )
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0)
            transcript = broker.close(build_reaped=True, validate=False)
            with mock.patch.object(contracts, "COMPILER_SOURCE_RECORD", transcript["source"]):
                contracts._validate_serialized_compiler_execution(transcript)
                missing = copy.deepcopy(transcript)
                missing["expected_recipe_keys"].append("never-issued")
                with self.assertRaisesRegex(contracts.EvidenceError, "omitted or duplicated"):
                    contracts._validate_serialized_compiler_execution(missing)
                unconsumed = copy.deepcopy(transcript)
                unconsumed["actions"][0]["state"] = "issued"
                unconsumed["actions"][0]["consumed_at_ns"] = None
                with self.assertRaisesRegex(contracts.EvidenceError, "issued/consumed manifest"):
                    contracts._validate_serialized_compiler_execution(unconsumed)

    def test_client_build_descendant_never_retains_sealed_compiler_inode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-fd-") as temporary:
            root = Path(temporary)
            compiler, broker, client, environment = self._broker(root, "/bin/sleep")
            process = subprocess.Popen(
                [broker.client_exec_path, "2"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            deadline = time.monotonic() + 3.0
            while (broker._active == 0 or broker._active_compiler_pid is None) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(broker.failure, None)
            self.assertIsNotNone(broker._active_compiler_pid)
            sealed_identity = os.fstat(compiler.fd)
            observed_pids = [process.pid]
            if broker._active_compiler_pid is not None:
                observed_pids.append(broker._active_compiler_pid)
            for observed_pid in observed_pids:
                for descriptor in Path(f"/proc/{observed_pid}/fd").iterdir():
                    details = descriptor.stat()
                    self.assertNotEqual((details.st_dev, details.st_ino), (sealed_identity.st_dev, sealed_identity.st_ino))
                    self.assertNotEqual(os.readlink(descriptor), f"/proc/{os.getpid()}/fd/{compiler.fd}")
            stdout, stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            self.assertTrue(containment.terminate_and_reap(process))
            broker.mark_build_reaped()
            transcript = broker.close(build_reaped=True, validate=False)
            self.assertEqual(transcript["request_count"], 1)

    def test_broker_services_actual_variable_invocation_stream_and_closes_from_observed_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-stream-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            command = "; ".join(
                f"{broker.client_exec_path} {argument}"
                for argument in ("1", "2", "3")
            )
            process = subprocess.Popen(
                ["/bin/sh", "-c", command], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            stdout, stderr = process.communicate(timeout=15.0)
            self.assertTrue(containment.terminate_and_reap(process))
            broker.mark_build_reaped()
            self.assertEqual(process.returncode, 0, repr(broker.failure))
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            transcript = broker.close(build_reaped=True, validate=False)
            self.assertEqual(transcript["request_count"], 3)
            self.assertEqual([event["sequence"] for event in transcript["events"]], [0, 1, 2])
            self.assertEqual(transcript["closure"]["last_sequence"], 2)
            with self.assertRaises(contracts.EvidenceError):
                contracts._validate_serialized_compiler_execution(transcript)

    def test_static_and_replayed_transcripts_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-promotion-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "1")
            self.assertEqual(process.returncode, 0)
            transcript = broker.close(build_reaped=True, validate=False)
            with self.assertRaises(contracts.EvidenceError):
                contracts.validate_compiler_execution_record(transcript)
            forged = dict(transcript)
            forged["events"] = [dict(transcript["events"][0]), dict(transcript["events"][0])]
            with self.assertRaises(contracts.EvidenceError):
                contracts._validate_serialized_compiler_execution(forged)

    def test_build_registration_race_is_barriered_until_the_parent_binds_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-registration-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            process = subprocess.Popen(
                [broker.client_exec_path, "1"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            time.sleep(0.1)
            containment = self._bind_build(broker, process)
            stdout, stderr = process.communicate(timeout=10.0)
            self.assertEqual((stdout, stderr), (b"", b""))
            self.assertTrue(containment.terminate_and_reap(process))
            broker.mark_build_reaped()
            self.assertIsNone(broker.failure)
            self.assertEqual(broker.close(build_reaped=True, validate=False)["request_count"], 1)

    def test_client_replacement_after_startup_cannot_change_sealed_launch_object(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-client-replacement-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            client.write_text(builder.COMPILER_CLIENT_TEMPLATE.replace("argv = sys.argv[1:]", "argv = ['1']"), encoding="utf-8")
            client.chmod(0o700)
            process = subprocess.Popen(
                [broker.client_exec_path, "1"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 0)
            self.assertIsNone(broker.failure)
            self.assertTrue(containment.terminate_and_reap(process))
            broker.mark_build_reaped()
            broker.close(build_reaped=True, validate=False)

    def test_compiler_injection_option_is_rejected_before_any_spawn(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-injection-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            process = subprocess.Popen(
                [broker.client_exec_path, "-include", "/tmp/unreviewed-header"], cwd=root, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, close_fds=True, pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 125)
            self.assertIsNotNone(broker.failure)
            self.assertTrue(containment.terminate_and_reap(process))

    def test_parent_owned_exact_actions_reject_unreviewed_option_values_and_generated_paths(self) -> None:
        rejected = (
            "-DATTACK=1", "-UOLD", "-Wno-everything", "-O0", "-std=c++23",
            "-fplugin=evil.so", "-mllvm", "999", "--offload-arch=gfx9999",
            "CMakeFiles/CompilerId/CMakeCCompilerId.c",
        )
        for argument in rejected:
            with self.subTest(argument=argument), tempfile.TemporaryDirectory(prefix="sllm-g1-graph-") as temporary:
                root = Path(temporary)
                trace = root / "compiler-trace"
                _compiler, broker, client, environment = self._broker(root, str(self._marker_compiler(root)))
                environment["SLLM_TEST_COMPILER_TRACE"] = str(trace)
                process, _stdout, _stderr = self._run_client(broker, client, root, environment, argument)
                self.assertEqual(process.returncode, 125)
                self.assertIsNotNone(broker.failure)
                self.assertEqual(broker._active_compiler_pid, None)
                self.assertFalse(trace.exists(), "unreviewed discovery/action launched a compiler before issuance")

    def test_cmake_semantic_discovery_never_executes_client_for_canonical_inherited_fds(self) -> None:
        for descriptor in (3, 9, 10, 29, 30, 300):
            with self.subTest(descriptor=descriptor), tempfile.TemporaryDirectory(prefix="sllm-g1-cmake-discovery-") as temporary:
                root = Path(temporary)
                client = root / "attacker-client"
                trace = root / "client-was-executed"
                client.write_text(f"#!/bin/sh\nprintf x > {trace}\n", encoding="utf-8")
                client.chmod(0o700)
                with self._descriptor_at(client, descriptor):
                    configure = self._run_cmake_semantic_discovery(root, trace, descriptor, str(descriptor))
                self.assertEqual(configure.returncode, 0, configure.stderr.decode("utf-8", "replace"))
                self.assertFalse(trace.exists(), "semantic CMake discovery executed the untrusted compiler client")
                cache = (root / "cmake-build" / "CMakeCache.txt").read_text(encoding="utf-8")
                self.assertIn("SLLM_SEMANTIC_G1_AUTHORITY:BOOL=ON", cache)

    def test_cmake_semantic_discovery_rejects_reserved_malformed_and_noncanonical_fds(self) -> None:
        rejected = ("0", "1", "2", "x", "10x", "03", "003", "+3", "-3")
        for descriptor_text in rejected:
            with self.subTest(descriptor=descriptor_text), tempfile.TemporaryDirectory(prefix="sllm-g1-cmake-discovery-reject-") as temporary:
                root = Path(temporary)
                client = root / "attacker-client"
                trace = root / "client-was-executed"
                client.write_text(f"#!/bin/sh\nprintf x > {trace}\n", encoding="utf-8")
                client.chmod(0o700)
                descriptor = os.open(client, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    configure = self._run_cmake_semantic_discovery(root, trace, descriptor, descriptor_text)
                finally:
                    os.close(descriptor)
                self.assertNotEqual(configure.returncode, 0, descriptor_text)
                self.assertFalse(trace.exists(), "malformed FD validation executed the untrusted compiler client")

    def test_helper_replacement_after_startup_cannot_change_sealed_exec(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-helper-replacement-") as temporary:
            root = Path(temporary)
            compiler, broker, _client, environment = self._broker(root)
            helper = root / "compiler-exec-helper"
            helper.chmod(0o700)
            helper.write_bytes(b"replacement")
            helper.chmod(0o555)
            process, stdout, stderr = self._run_client(broker, root / "compiler-client.py", root, environment, "1")
            self.assertEqual(process.returncode, 0)
            self.assertEqual((stdout, stderr), (b"", b""))
            self.assertEqual(broker.events[0]["compiler"]["exec_identity"]["sealed_dev"], os.fstat(compiler.fd).st_dev)

    def test_compiler_nonzero_status_is_returned_without_protocol_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-status-") as temporary:
            root = Path(temporary)
            failing_compiler = root / "failing-compiler"
            completed = subprocess.run(
                ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(failing_compiler), "-"],
                input=b"#include <unistd.h>\nint main(){sleep(1);return 7;}\n",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            failing_compiler.chmod(0o555)
            _compiler, broker, client, environment = self._broker(root, str(failing_compiler))
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "3")
            self.assertEqual(process.returncode, 7)
            event = broker.events[0]
            self.assertEqual(event["compiler"]["status"], "failed")
            self.assertEqual(event["compiler"]["exit_code"], 7)
            transcript = broker.close(build_reaped=True, validate=False)
            self.assertEqual(transcript["events"][0]["compiler"]["status"], "failed")

    def test_compiler_signal_status_is_returned_without_protocol_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-crash-") as temporary:
            root = Path(temporary)
            crashing_compiler = root / "crashing-compiler"
            completed = subprocess.run(
                ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(crashing_compiler), "-"],
                input=b"#include <unistd.h>\n#include <signal.h>\nint main(){sleep(1);raise(SIGSEGV);}\n",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            crashing_compiler.chmod(0o555)
            _compiler, broker, client, environment = self._broker(root, str(crashing_compiler))
            process, _stdout, _stderr = self._run_client(broker, client, root, environment, "3")
            self.assertEqual(process.returncode, -signal.SIGSEGV)
            event = broker.events[0]
            self.assertEqual(event["compiler"]["status"], "failed")
            self.assertEqual(event["compiler"]["exit_code"], -signal.SIGSEGV)
            self.assertTrue(event["compiler"]["crashed"])

    def test_late_registration_is_rejected_once_closing_starts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-late-bind-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            process = subprocess.Popen(["/bin/true"], cwd=root, start_new_session=True)
            containment = builder.runner.LinuxContainment.begin()
            containment.bind_root(process.pid, process.pid)
            with broker._lifecycle:
                broker._state = "closing"
            with self.assertRaises(builder.BuilderError):
                broker.bind_build(process.pid, os.getpgid(process.pid), process=process, containment=containment)
            process.wait(timeout=5.0)
            self.assertTrue(containment.terminate_and_reap(process))
            broker.abort()

    def test_client_escape_is_rejected_and_broker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-escape-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            process = subprocess.Popen(
                [broker.client_exec_path, "--version"], cwd="/tmp", env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 125)
            self.assertIsNotNone(broker.failure)
            with self.assertRaises(builder.BuilderError):
                broker.close(build_reaped=True)

    def test_client_death_fails_closed_and_reaps_the_broker_compiler(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-client-death-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root, "/bin/sleep")
            process = subprocess.Popen(
                [broker.client_exec_path, "2"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            deadline = time.monotonic() + 3.0
            while broker._active_compiler_pid is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker._active_compiler_pid)
            process.kill()
            process.communicate(timeout=5.0)
            deadline = time.monotonic() + 5.0
            while broker.failure is None and broker._active_compiler_pid is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker.failure)
            self.assertIsNone(broker._active_compiler_pid)
            with self.assertRaises(builder.BuilderError):
                broker.close(build_reaped=True)

    def test_broker_death_fails_closed_and_does_not_leave_compiler_alive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-death-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root, "/bin/sleep")
            process = subprocess.Popen(
                [broker.client_exec_path, "2"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            deadline = time.monotonic() + 1.0
            while broker._active_compiler_pid is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker._active_compiler_pid)
            broker.abort()
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertIsNotNone(process.returncode)
            self.assertIsNotNone(broker.failure)
            self.assertIsNone(broker._active_compiler_pid)

    def test_actual_broker_process_death_fails_client_and_reaps_compiler(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-real-broker-death-") as temporary:
            root = Path(temporary)
            client = root / "compiler-client.py"
            client.write_text(builder.COMPILER_CLIENT_TEMPLATE, encoding="utf-8")
            client.chmod(0o700)
            child_code = f'''
import json, os, pathlib, subprocess, sys, time
sys.path.insert(0, {str(ROOT / "ci/tools")!r})
import build_rmsnorm_g1_runtime as builder
import validate_rmsnorm_g1_contracts as contracts
root = pathlib.Path({str(root)!r})
helper = root / "compiler-exec-helper"
done = subprocess.run(
    ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti", "-o", str(helper), "-"],
    input=builder.COMPILER_EXEC_HELPER_SOURCE.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
)
if done.returncode != 0:
    raise SystemExit(91)
helper.chmod(0o555)
compiler = contracts.snapshot_file(pathlib.Path("/bin/sleep"), None, "real broker death compiler")
broker = builder.CompilerBroker(compiler=compiler, client_path=root / "compiler-client.py", exec_helper=helper, allowed_roots=(root,), action_recipes={{"death": {{"argv": ["2"], "cwd": str(root), "inputs": [], "implicit": [], "response_files": [], "outputs": []}}}})
broker.start()
environment = {{"PATH": "/usr/bin:/bin", "HOME": "/tmp", **broker.environment()}}
client_process = subprocess.Popen(
    [broker.client_exec_path, "2"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
    pass_fds=broker.child_pass_fds(),
)
containment = builder.runner.LinuxContainment.begin()
containment.bind_root(client_process.pid, client_process.pid)
broker.bind_build(client_process.pid, os.getpgid(client_process.pid), environment, process=client_process, containment=containment)
deadline = time.monotonic() + 5.0
while broker._active_compiler_pid is None and time.monotonic() < deadline:
    time.sleep(0.01)
if broker._active_compiler_pid is None:
    raise SystemExit(92)
print(json.dumps({{"client_pid": client_process.pid, "compiler_pid": broker._active_compiler_pid}}), flush=True)
time.sleep(60.0)
'''
            controller = subprocess.Popen(
                [sys.executable, "-B", "-c", child_code], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                close_fds=True,
            )
            containment = builder.runner.LinuxContainment.begin()
            containment.bind_root(controller.pid, controller.pid)
            try:
                line = controller.stdout.readline() if controller.stdout is not None else b""
                self.assertTrue(line)
                identities = json.loads(line.decode("utf-8"))
                client_pid = int(identities["client_pid"])
                compiler_pid = int(identities["compiler_pid"])
                containment.observe()
                controller.kill()
                controller.wait(timeout=5.0)
                deadline = time.monotonic() + 5.0
                client_status = None
                while time.monotonic() < deadline:
                    waited, status = os.waitpid(client_pid, os.WNOHANG)
                    if waited == client_pid:
                        client_status = os.waitstatus_to_exitcode(status)
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(client_status)
                self.assertNotEqual(client_status, 0)
                self.assertTrue(containment.terminate_and_reap(controller))
                self.assertIsNone(builder._process_facts(compiler_pid))
            finally:
                builder.runner.close_process_streams(controller)
                if not containment.subreaper_restored:
                    self.assertTrue(containment.terminate_and_reap(controller))

    def test_post_close_client_request_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-post-close-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            process = subprocess.Popen(["/bin/true"], cwd=root, env=environment, start_new_session=True, close_fds=True)
            containment = self._bind_build(broker, process)
            process.wait(timeout=5.0)
            self.assertTrue(containment.terminate_and_reap(process))
            broker.mark_build_reaped()
            broker.close(build_reaped=True, validate=False)
            client_process = subprocess.run(
                [str(client), "--version"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(client_process.returncode, 125)

    def test_forged_hmac_and_unexpected_ancillary_fd_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-forgery-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            process = subprocess.Popen(["/bin/sleep", "5"], start_new_session=True)
            self._bind_build(broker, process)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.addCleanup(connection.close)
            connection.connect(str(broker.socket_path))
            connection.send(b"{}")
            deadline = time.monotonic() + 3.0
            while broker.failure is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker.failure)

    def test_scm_rights_rejection_closes_all_received_descriptors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-rights-budget-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            process = subprocess.Popen(["/bin/sleep", "5"], start_new_session=True)
            self._bind_build(broker, process)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.addCleanup(connection.close)
            connection.connect(str(broker.socket_path))
            descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(8)]
            before = len(list(Path("/proc/self/fd").iterdir()))
            try:
                connection.sendmsg([b"{}"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{len(descriptors)}i", *descriptors))])
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)
            deadline = time.monotonic() + 3.0
            while broker.failure is None and time.monotonic() < deadline:
                time.sleep(0.01)
            after = len(list(Path("/proc/self/fd").iterdir()))
            self.assertIsNotNone(broker.failure)
            self.assertLessEqual(after, before + 2)

        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-ancillary-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            process = subprocess.Popen(["/bin/sleep", "5"], start_new_session=True)
            self._bind_build(broker, process)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.addCleanup(connection.close)
            connection.connect(str(broker.socket_path))
            read_fd = os.open("/dev/null", os.O_RDONLY)
            try:
                connection.sendmsg([b"{}"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", read_fd))])
            finally:
                os.close(read_fd)
            deadline = time.monotonic() + 3.0
            while broker.failure is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker.failure)

    def test_client_ancillary_cleanup_handles_rights_truncation(self) -> None:
        for count in (1, 64):
            with self.subTest(count=count), tempfile.TemporaryDirectory(prefix="sllm-g1-client-rights-") as temporary:
                root = Path(temporary)
                client = root / "compiler-client.py"
                client.write_text(builder.COMPILER_CLIENT_TEMPLATE, encoding="utf-8")
                client.chmod(0o700)
                socket_path = root / "fake-broker.sock"
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                listener.bind(str(socket_path))
                listener.listen(1)
                token = "ab" * 32
                environment = {
                    "PATH": "/usr/bin:/bin", "HOME": "/tmp",
                    builder.COMPILER_BROKER_SOCKET_ENV: str(socket_path),
                    builder.COMPILER_BROKER_TOKEN_ENV: token,
                    builder.COMPILER_BROKER_SESSION_ENV: "cd" * 32,
                }
                process = subprocess.Popen(
                    [str(client), "--version"], cwd=root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                connection, _address = listener.accept()
                descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(count)]
                before = len(list(Path("/proc/self/fd").iterdir()))
                try:
                    connection.sendmsg([b"{}"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{count}i", *descriptors))])
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)
                    connection.close()
                    listener.close()
                stdout, stderr = process.communicate(timeout=5.0)
                self.assertEqual(process.returncode, 125)
                self.assertEqual((stdout, stderr), (b"", b""))
                after = len(list(Path("/proc/self/fd").iterdir()))
                self.assertLessEqual(after, before + 2)

    def test_compiler_environment_injection_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-environment-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root)
            forged_environment = {**environment, "LD_PRELOAD": "/tmp/does-not-exist.so"}
            process = subprocess.Popen(
                [broker.client_exec_path, "--version"], cwd=root, env=forged_environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 125)
            self.assertIsNotNone(broker.failure)

    def test_truncated_or_oversized_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-frame-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            process = subprocess.Popen(["/bin/sleep", "5"], start_new_session=True)
            self._bind_build(broker, process)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.addCleanup(connection.close)
            connection.connect(str(broker.socket_path))
            with mock.patch.object(builder, "COMPILER_BROKER_MAX_FRAME", 1024):
                connection.send(b"x" * 2048)
            deadline = time.monotonic() + 3.0
            while broker.failure is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(broker.failure)

    def test_replayed_authenticated_nonce_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-replay-") as temporary:
            root = Path(temporary)
            client = root / "compiler-client.py"
            replay_source = builder.COMPILER_CLIENT_TEMPLATE.replace(
                "        response, response_frame_sha256 = exchange(socket_path, token, request)\n",
                "        response, response_frame_sha256 = exchange(socket_path, token, request)\n"
                "        exchange(socket_path, token, request)\n",
            )
            self.assertNotEqual(replay_source, builder.COMPILER_CLIENT_TEMPLATE)
            client.write_text(replay_source, encoding="utf-8")
            client.chmod(0o700)
            compiler = contracts.snapshot_file(Path("/bin/sleep"), None, "replay test compiler")
            helper = root / "compiler-exec-helper"
            completed = subprocess.run(
                ["/usr/bin/c++", "-x", "c++", "-std=c++17", "-O2", "-o", str(helper), "-"],
                input=builder.COMPILER_EXEC_HELPER_SOURCE.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            helper.chmod(0o555)
            broker = builder.CompilerBroker(compiler=compiler, client_path=client, exec_helper=helper, allowed_roots=(root,), action_recipes={"nonce": {"argv": ["1"], "cwd": str(root), "inputs": [], "implicit": [], "response_files": [], "outputs": []}})
            broker.start()
            self.addCleanup(broker.abort)
            self.addCleanup(compiler.close)
            environment = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", **broker.environment()}
            process = subprocess.Popen(
                [broker.client_exec_path, "1"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                pass_fds=broker.child_pass_fds(),
            )
            containment = self._bind_build(broker, process)
            _stdout, _stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 125)
            self.assertIsNotNone(broker.failure)

    def test_compiler_timeout_cleans_up_pid_bound_child(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-timeout-") as temporary:
            root = Path(temporary)
            _compiler, broker, client, environment = self._broker(root, "/bin/sleep")
            with mock.patch.object(builder, "COMPILER_BROKER_TIMEOUT_SECONDS", 0.1):
                process, _stdout, _stderr = self._run_client(broker, client, root, environment, "5")
            self.assertEqual(process.returncode, 125)
            self.assertIsNotNone(broker.failure)
            self.assertEqual(broker._active, 0)

    def test_build_popen_failure_aborts_broker_and_restores_containment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-broker-popen-") as temporary:
            root = Path(temporary)
            _compiler, broker, _client, _environment = self._broker(root)
            with mock.patch.object(builder.subprocess, "Popen", side_effect=OSError("injected Popen failure")):
                with self.assertRaises(builder.BuilderError):
                    builder._run(
                        ["/bin/true"], cwd=root, env={"PATH": "/usr/bin:/bin"}, timeout=1.0,
                        compiler_source=broker.source, broker=broker,
                    )
            self.assertFalse(broker._thread.is_alive())
            self.assertFalse(broker.socket_path.exists())

    def test_arbitrary_build_command_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-build-exploit-") as temporary:
            with self.assertRaisesRegex(builder.BuilderError, "FAIL-CLOSED"):
                builder._run(["/bin/bash", "-c", "echo forged"], cwd=Path(temporary), env={"PATH": "/usr/bin:/bin"}, timeout=1.0)

    def test_static_contract_requires_parent_owned_broker(self) -> None:
        contracts.validate_compiler_execution_contract(ROOT)
        builder_source = (ROOT / "ci/tools/build_rmsnorm_g1_runtime.py").read_text(encoding="utf-8")
        self.assertIn("COMPILER_BROKER_AVAILABLE = True", builder_source)
        self.assertIn("execveat", builder_source)
        self.assertNotIn("sealed-memfd-wrapper-events-v3", builder_source)
        self.assertNotIn("EXPECTED_COMPILER_EVENT_PLAN", builder_source)


if __name__ == "__main__":
    unittest.main()
