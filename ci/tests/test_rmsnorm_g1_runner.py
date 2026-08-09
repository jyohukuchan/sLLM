"""Host-safe production-path checks for the semantic G1 fixed worker."""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import run_rmsnorm_g1_runtime as runner  # noqa: E402
import validate_rmsnorm_g1_contracts as contracts  # noqa: E402


def _fixed(value: str) -> bytes:
    raw = value.encode("ascii")
    return raw + b"\0" + b"\0" * (63 - len(raw))


def raw_response(
    shape: tuple[int, ...] = (1, 3),
    *,
    target: str = "gfx1030",
    device_index: int = 0,
    allocation_count: int = 3,
    copy_count: int = 3,
    kernel_count: int = 1,
) -> bytes:
    extents = (*shape, *(0 for _ in range(8 - len(shape))))
    elements = 1
    for value in shape:
        elements *= value
    rows = elements // shape[-1]
    output = b"\0\0" * elements
    data = b"".join((
        runner.OUTPUT_MAGIC,
        struct.pack("<II", runner.OUTPUT_PROTOCOL_VERSION, runner.OUTPUT_HEADER_BYTES),
        struct.pack("<II", len(shape), 0),
        struct.pack("<8Q", *extents),
        struct.pack("<QQQ", elements, shape[-1], rows),
        struct.pack("<II", struct.unpack("<I", struct.pack("<f", 1.0e-5))[0], 0),
        struct.pack("<II", device_index, 1),
        struct.pack("<QIIII", 7, 1, 1, 256, rows),
        struct.pack("<II", 0, 0),
        struct.pack("<IIIIIII", 1, 1, 3, 2, 1, 3, 0),
        struct.pack("<IIII", allocation_count, copy_count, kernel_count, 0),
        struct.pack("<QQQ", 0, 0, 0),
        _fixed(runner.KERNEL_SYMBOL),
        _fixed(runner.DEVICE_SYMBOL),
        _fixed(target),
        struct.pack("<Q", len(output)),
        output,
    ))
    assert len(data) == runner.OUTPUT_HEADER_BYTES + len(output)
    return data


class SemanticG1RunnerTests(unittest.TestCase):
    def _reap(self, process: subprocess.Popen[bytes], pgid: int) -> None:
        self.assertTrue(runner.cleanup_process_group(process, pgid))
        runner.close_process_streams(process)

    def test_physical_hip_index_is_matrix_bound_not_topology_hard_coded(self) -> None:
        for index in (0, 1, 17):
            expected = str(index)
            self.assertEqual(runner.semantic_runtime_environment(index)["HIP_VISIBLE_DEVICES"], expected)
            self.assertEqual(contracts.semantic_runtime_environment(index)["HIP_VISIBLE_DEVICES"], expected)
        for invalid in (-1, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(runner.RunnerError):
                    runner.semantic_runtime_environment(invalid)  # type: ignore[arg-type]
                with self.assertRaises(contracts.EvidenceError):
                    contracts.semantic_runtime_environment(invalid)  # type: ignore[arg-type]

    def test_v2_raw_protocol_requires_exact_semantic_resource_counts(self) -> None:
        data = raw_response((2, 17))
        parsed = runner.parse_response(
            data,
            expected_target="gfx1030",
            expected_device_index=0,
            expected_shape=(2, 17),
            expected_epsilon=1.0e-5,
        )
        self.assertEqual(parsed["resource_counts"], {
            "allocation_count": 3,
            "copy_count": 3,
            "dispatch_count": 1,
            "kernel_count": 1,
        })
        with self.assertRaises(runner.RunnerError):
            runner.parse_response(
                raw_response(allocation_count=4),
                expected_target="gfx1030",
                expected_device_index=0,
                expected_shape=(1, 3),
                expected_epsilon=1.0e-5,
            )

    def test_controller_worker_ancillary_cleanup_covers_unknown_before_rights_and_ctrunc(self) -> None:
        def send_bad(sender: socket.socket, count: int, duplicate_credentials: bool) -> list[int]:
            descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(count)]
            credential = struct.pack("3i", os.getpid(), os.getuid(), os.getgid())
            control = []
            if duplicate_credentials:
                control.extend([
                    (socket.SOL_SOCKET, socket.SCM_CREDENTIALS, credential),
                    (socket.SOL_SOCKET, socket.SCM_CREDENTIALS, credential),
                ])
            control.append((socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{count}i", *descriptors)))
            sender.sendmsg([b"{}"], control)
            return descriptors

        for receive in (contracts.ipc_recv, runner._ipc_recv):
            with self.subTest(receive=receive.__name__):
                left, right = contracts.controller_socketpair()
                before = len(list(Path("/proc/self/fd").iterdir()))
                descriptors = send_bad(right, 64, True)
                try:
                    with self.assertRaises((contracts.EvidenceError, runner.RunnerError)):
                        receive(left)
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)
                    left.close()
                    right.close()
                after = len(list(Path("/proc/self/fd").iterdir()))
                self.assertLessEqual(after, before + 2)

        class TruncatedSocket:
            def recvmsg(self, *_args: object) -> tuple[bytes, list[tuple[int, int, bytes]], int, None]:
                return b"{}", [], socket.MSG_CTRUNC, None

        with self.assertRaises(contracts.EvidenceError):
            contracts.ipc_recv(TruncatedSocket())  # type: ignore[arg-type]
        with self.assertRaises(runner.RunnerError):
            runner._ipc_recv(TruncatedSocket())  # type: ignore[arg-type]

    def test_review10_case_set_binds_epsilon_and_explicit_nan_inf_inputs(self) -> None:
        expected = {case["id"]: case for case in contracts.EXPECTED_CASES}
        self.assertEqual(len(expected), 15)
        self.assertEqual(expected["r1-n2560"]["n"], 2560)
        self.assertEqual(contracts.MODEL_EPSILON, 1.0e-6)
        for source in ("activation", "raw_scale"):
            for case_id, value in ((f"r1-n2560{'-raw-scale' if source == 'raw_scale' else ''}-nan", float("nan")), (f"r1-n2560{'-raw-scale' if source == 'raw_scale' else ''}-posinf", float("inf")), (f"r1-n2560{'-raw-scale' if source == 'raw_scale' else ''}-neginf", float("-inf"))):
                self.assertEqual(expected[case_id]["nonfinite_input"], source)
                bits = runner._f32_to_bf16(value)
                activation = struct.pack("<H", bits) + b"\0\0" * (2560 - 1) if source == "activation" else b"\0\0" * 2560
                scale = struct.pack("<H", bits) + b"\0\0" * (2560 - 1) if source == "raw_scale" else b"\0\0" * 2560
                output = runner.independent_rmsnorm_oracle(activation, scale, 1, 2560, contracts.MODEL_EPSILON)
                values = [runner._bf16_to_f32(item[0]) for item in struct.iter_unpack("<H", output)]
                self.assertTrue(any(runner.math.isnan(item) for item in values), case_id)

    def test_recursive_loader_proof_matches_live_rpath_and_runpath_semantics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-loader-semantics-") as temporary:
            root = Path(temporary)
            for name, extra_linker_flag, expected_kind in (
                ("rpath", "-Wl,--disable-new-dtags", "rpath"),
                ("runpath", "", "runpath"),
            ):
                case_root = root / name
                library_dir = case_root / "lib"
                library_dir.mkdir(parents=True)
                library_source = case_root / "fixture.c"
                library_source.write_text("int sllm_fixture(void) { return 42; }\n", encoding="ascii")
                library = library_dir / "libsllm_fixture.so"
                compile_library = subprocess.run(
                    ["/usr/bin/cc", "-shared", "-fPIC", "-Wl,-soname,libsllm_fixture.so", "-o", str(library), str(library_source)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                )
                self.assertEqual(compile_library.returncode, 0, compile_library.stderr.decode("utf-8", "replace"))
                executable_source = case_root / "main.c"
                executable_source.write_text("extern int sllm_fixture(void); int main(void) { return sllm_fixture() != 42; }\n", encoding="ascii")
                executable = case_root / "fixture"
                link = [
                    "/usr/bin/cc", "-Wl,--no-as-needed", "-L", str(library_dir), "-o", str(executable),
                    str(executable_source), "-lsllm_fixture", "-Wl,-rpath,$ORIGIN/lib",
                ]
                if extra_linker_flag:
                    link.append(extra_linker_flag)
                compile_executable = subprocess.run(link, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                self.assertEqual(compile_executable.returncode, 0, compile_executable.stderr.decode("utf-8", "replace"))
                closure = contracts.runtime_dependency_closure(executable)
                root_object = next(item for item in closure["objects"] if item["record"]["resolved_path"] == str(executable.resolve()))
                fixture_edge = next(edge for edge in root_object["needed"] if edge["name"] == "libsllm_fixture.so")
                self.assertEqual(fixture_edge["search_kind"], expected_kind)
                self.assertEqual(fixture_edge["resolved_path"], str(library.resolve()))
                retained_descriptors = []
                retained = {}
                try:
                    for item in closure["objects"]:
                        record = item["record"]
                        descriptor = contracts.snapshot_file(Path(record["path"]), record, f"loader semantics {name}")
                        retained_descriptors.append(descriptor)
                        retained[str(record["resolved_path"])] = (descriptor.record, descriptor.fd)
                    executable_fd = os.open(executable, os.O_RDONLY | os.O_CLOEXEC)
                    try:
                        literal_loader = contracts.elf_interpreter_path(executable_fd)
                    finally:
                        os.close(executable_fd)
                    self.assertIsNotNone(literal_loader)
                    loader_object = next(item for item in closure["objects"] if item["record"]["resolved_path"] == str(root_object["interpreter"]))
                    loader_record = {
                        **loader_object["record"],
                        "path": str(literal_loader),
                    }
                    loader_descriptor = contracts.snapshot_file(
                        literal_loader,
                        loader_record,
                        f"literal loader semantics {name}",
                    )
                    retained_descriptors.append(loader_descriptor)
                    retained[str(loader_descriptor.record["resolved_path"])] = (loader_descriptor.record, loader_descriptor.fd)
                    contracts.validate_runtime_dependency_closure(
                        retained, closure, root_path=str(executable.resolve()), loader_path=str(root_object["interpreter"]),
                    )
                finally:
                    for descriptor in retained_descriptors:
                        descriptor.close()

    def test_popen_failure_restores_subreaper_and_limits_are_hard_launch_contract(self) -> None:
        containment = runner.LinuxContainment.begin()
        self.assertFalse(containment.tracked)
        with self.assertRaises(FileNotFoundError):
            subprocess.Popen(["/definitely-missing-semantic-g1-executable"], close_fds=True)
        self.assertTrue(containment.restore_after_launch_failure())
        self.assertTrue(containment.subreaper_restored)
        self.assertEqual(runner.PROCESS_LIMITER, "/usr/bin/prlimit")
        self.assertEqual(runner.PROCESS_ADDRESS_LIMIT_BYTES, 64 * 1024 * 1024 * 1024)
        self.assertGreater(runner.PROCESS_ADDRESS_LIMIT_BYTES, runner.MAX_RUNTIME_RSS_BYTES)
        self.assertEqual(runner.PROCESS_COUNT_LIMIT, 4096)

    def test_actual_child_fd_audit_allows_internal_fds_and_rejects_unrelated_parent_fd(self) -> None:
        sealed = contracts.snapshot_bytes(b"expected inherited bytes", logical_path="/controller/expected", label="expected")
        self.addCleanup(sealed.close)
        command = ["/usr/bin/python3", "-c", "import time; time.sleep(30)"]
        parent_identities = runner._parent_open_identities()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=(sealed.fd,),
        )
        try:
            binding = contracts.process_binding(process.pid)
            runner.audit_child_fd_inheritance(
                process.pid,
                (sealed.fd,),
                expected_binding=binding,
                parent_identities=parent_identities,
            )
        finally:
            self._reap(process, process.pid)

        with tempfile.TemporaryDirectory(prefix="sllm-g1-audit-") as temporary:
            unrelated_path = Path(temporary) / "unrelated"
            unrelated_path.write_bytes(b"not an inherited controller capability")
            unrelated = os.open(unrelated_path, os.O_RDONLY | os.O_CLOEXEC)
            self.addCleanup(lambda: os.close(unrelated) if unrelated >= 0 else None)
            parent_identities = runner._parent_open_identities()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(sealed.fd, unrelated),
            )
            try:
                with self.assertRaises(runner.RunnerError):
                    runner.audit_child_fd_inheritance(
                        process.pid,
                        (sealed.fd,),
                        expected_binding=contracts.process_binding(process.pid),
                        forbidden_descriptors=(unrelated,),
                        parent_identities=parent_identities,
                    )
            finally:
                self._reap(process, process.pid)
                os.close(unrelated)
                unrelated = -1

    def test_descendant_that_retains_pipes_is_killed_after_direct_child_exit(self) -> None:
        process = subprocess.Popen(
            ["/bin/sh", "-c", "(sleep 30) & exit 0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.bounded_exchange(process, b"", 0.1)
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            self._reap(process, process.pid)
        self.assertEqual(runner._group_members(process.pid), [])

    def test_rss_limit_and_setsid_escape_use_pidfd_subreaper_containment(self) -> None:
        prior_subreaper = runner._child_subreaper_state()
        containment = runner.LinuxContainment.begin()
        process = subprocess.Popen(
            ["/usr/bin/python3", "-c", "import os,time; os.system('setsid sleep 30 &'); x=bytearray(32 * 1024 * 1024); time.sleep(30)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        containment.bind_root(process.pid, process.pid)
        try:
            with self.assertRaises(runner.RunnerError):
                runner.bounded_exchange(process, b"", 2.0, containment=containment, rss_limit_bytes=1024)
        finally:
            self.assertTrue(containment.terminate_and_reap(process))
            runner.close_process_streams(process)
        self.assertEqual(runner._child_subreaper_state(), prior_subreaper)
        self.assertEqual(runner._group_members(process.pid), [])

    def test_worker_surface_has_no_report_or_injected_execution_authority(self) -> None:
        source = (ROOT / contracts.RUNNER_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertIn("SOCK_SEQPACKET", (ROOT / "ci/tools/validate_rmsnorm_g1_contracts.py").read_text(encoding="utf-8"))
        self.assertIn("start_new_session=True", source)
        self.assertIn("audit_child_fd_inheritance", source)
        self.assertIn("LinuxContainment", source)
        self.assertIn("pidfd_send_signal", source)
        self.assertNotIn("runner_script", source)
        self.assertNotIn("popen_factory", source)
        self.assertNotIn("aggregate_results", source)
        self.assertNotIn("validate_rmsnorm_g1_contracts", source)


if __name__ == "__main__":
    unittest.main()
