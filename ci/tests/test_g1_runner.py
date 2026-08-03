#!/usr/bin/env python3
"""Host-only mocked positive and negative tests for the trusted-local G1 runner."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import run_g1_evidence as runner  # noqa: E402
from common import ContractError, canonical_bytes, sha256_file, sha256_json  # noqa: E402
from validate_g0_contracts import AMD_SMI_EXECUTABLE, AMD_SMI_LIST_COMMAND  # noqa: E402
from validate_g1_contracts import (  # noqa: E402
    BINARY_NAME,
    EXPECTED_SIZES,
    METADATA_NAME,
    _manifest_hashes,
    inspect_g1_runtime_artifact,
    validate_g1_matrix,
    validate_artifact_metadata,
    validate_schema,
    row_by_id,
)

KERNEL_SYMBOL = "_ZN12_GLOBAL__N_118evidence_transformEPKhPhm"


def host_readobj_fixture() -> str:
    return """File: final
Format: elf64-x86-64
Arch: x86_64
Sections [
  Section {
    Name: .text (1)
    Size: 64
  }
  Section {
    Name: .hip_fatbin (2)
    Size: 128
  }
]
"""


def device_readobj_fixture(target: str) -> str:
    flags = {"gfx1030": "36", "gfx1201": "4e"}[target]
    return f"""File: device-code-object.elf
Format: elf64-amdgpu
Arch: amdgcn
ElfHeader {{
  Ident {{
    ABIVersion: 4
  }}
  Flags [ (0x{flags})
  ]
}}
Sections [
  Section {{
    Name: .text (1)
    Size: 64
  }}
]
Symbols [
  Symbol {{
    Name:  (0)
    Type: None (0x0)
    Section: Undefined (0x0)
  }}
  Symbol {{
    Name: {KERNEL_SYMBOL} (1)
    Type: Function (0x2)
    Section: .text (0x1)
  }}
  Symbol {{
    Name: {KERNEL_SYMBOL}.kd (2)
    Type: Object (0x1)
    Section: .rodata (0x2)
  }}
]
NoteSections [
  NoteSection {{
    .name: {KERNEL_SYMBOL}
    .symbol: {KERNEL_SYMBOL}.kd
    .wavefront_size: 32
  }}
]
amdhsa.target:   amdgcn-amd-amdhsa--{target}
"""


class FixtureToolRunner:
    """Deterministic pinned-tool output for the current code-object contract."""

    def __init__(self, target: str) -> None:
        self.target = target

    def __call__(self, argv, **_kwargs):
        command = tuple(str(item) for item in argv)
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-readobj":
            output = device_readobj_fixture(self.target) if command[-1].endswith("device-code-object.elf") else host_readobj_fixture()
            return type("Result", (), {"returncode": 0, "stdout": output.encode(), "stderr": b""})()
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-objcopy":
            destination = next(item.split("=", 2)[2] for item in command if item.startswith("--dump-section=.hip_fatbin="))
            Path(destination).write_bytes(b"deterministic-fatbin")
            return type("Result", (), {"returncode": 0, "stdout": b"", "stderr": b""})()
        if command[0] == "/opt/rocm/lib/llvm/bin/clang-offload-bundler" and "--list" in command:
            bundles = [f"hipv4-amdgcn-amd-amdhsa--{self.target}", "host-x86_64-unknown-linux-gnu-"]
            return type("Result", (), {"returncode": 0, "stdout": ("\n".join(bundles) + "\n").encode(), "stderr": b""})()
        if command[0] == "/opt/rocm/lib/llvm/bin/clang-offload-bundler" and "--unbundle" in command:
            destination = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
            Path(destination).write_bytes(b"deterministic-device-code-object")
            return type("Result", (), {"returncode": 0, "stdout": b"", "stderr": b""})()
        raise AssertionError(f"unexpected inspector command: {command}")


def write_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def health(row: dict[str, object], timestamp: str) -> dict[str, object]:
    return {
        "available": True,
        "reliable": True,
        "observed_at": timestamp,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1",
        "facts": {
            "device_state": "active",
            "amdgpu_driver_bound": True,
            "runtime_status": "active",
            "ras_uncorrectable_count": 17,
            "sysfs_ras_uncorrectable_count": 17,
            "temperature_c": 47.5,
        },
    }


def processes(row: dict[str, object], timestamp: str) -> dict[str, object]:
    return {
        "available": True,
        "reliable": True,
        "observed_at": timestamp,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1",
        "gpu_processes": [],
        "residual_runner_children": [],
    }


@contextmanager
def unlocked_host_lock(_path: Path):
    yield


class G1RunnerFixture:
    def __init__(self, target: str = "gfx1030") -> None:
        self.stage = Path(tempfile.mkdtemp(prefix="ullm-g1-stage-"))
        self.output_root = Path(tempfile.mkdtemp(prefix="ullm-g1-output-"))
        self.matrix = validate_g1_matrix(ROOT)
        self.row = row_by_id(self.matrix, f"g1-{target}")
        self.row_id = self.row["row_id"]
        self.row_dir = self.stage / self.row_id
        self.row_dir.mkdir()
        self.artifact = self.stage / "target" / "release" / BINARY_NAME
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"dedicated-rust-evidence-binary\n")
        self.artifact.chmod(0o700)
        write_sidecar(self.artifact)
        self.tool_runner = FixtureToolRunner(target)
        inspection = inspect_g1_runtime_artifact(self.artifact, target, tool_runner=self.tool_runner)
        manifest_hashes = _manifest_hashes(ROOT)
        self.identity = {
            "run_id": "unit-g1",
            "run_attempt": 1,
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40,
        }
        self.metadata_path = self.row_dir / METADATA_NAME
        self.metadata = {
            "schema_version": "g1-runtime-artifact-v1",
            "metadata_id": f"g1-runtime-artifact-{target}",
            "row_id": self.row_id,
            "target": target,
            "candidate": {
                "reviewed_sha": self.identity["reviewed_sha"],
                "tested_sha": self.identity["tested_sha"],
                "workflow_sha": self.identity["workflow_sha"],
                "git_tree_oid": self.identity["git_tree_oid"],
                "worktree_clean": True,
                "revision_input": "full-sha",
            },
            "toolchain_id": "rocm-7.14.0",
            "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
            "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
            "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
            "gpu": {
                "bdf": self.row["bdf"],
                "uuid": self.row["uuid"],
                "target": target,
            },
            "artifact": {
                "path": str(self.artifact),
                "size_bytes": self.artifact.stat().st_size,
                "sha256": sha256_file(self.artifact),
                "sidecar_sha256": sha256_file(self.artifact.with_name(BINARY_NAME + ".sha256")),
                "kind": "dedicated-rust-evidence-binary",
            },
            "observed": inspection["observed"],
            "device_code_sha256": inspection["device_code_sha256"],
            "scope": {
                "model_used": False,
                "cpu_fallback_allowed": False,
                "cpu_fallback_used": False,
                "binary_command": list(runner.COMMAND),
            },
        }
        self.rewrite_metadata()

    def rewrite_metadata(self) -> None:
        self.metadata_path.write_bytes(canonical_bytes(self.metadata))
        write_sidecar(self.metadata_path)

    def runtime_binding(self) -> dict[str, object]:
        return {
            "rocm_root": "/opt/rocm",
            "rocm_release": "7.14.0",
            "path": runner.PINNED_PATH,
            "ld_library_path": runner.PINNED_LD_LIBRARY_PATH,
            "observation_method": "proc-pid-maps-poll-v1",
            "required_libraries": ["libamdhip64.so.7", "libhsa-runtime64.so.1"],
            "loaded_libraries": {
                "libamdhip64.so.7": "/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000",
                "libhsa-runtime64.so.1": "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0",
            },
            "inherited_loader_environment": False,
        }

    def output(self) -> Path:
        return self.output_root / self.row_id

    def argv(self, *, output: Path | None = None) -> list[str]:
        return [
            "--row", self.row_id,
            "--runtime-metadata", str(self.metadata_path),
            "--output-dir", str(output or self.output()),
            "--trusted-local",
            "--run-id", self.identity["run_id"],
            "--run-attempt", str(self.identity["run_attempt"]),
            "--reviewed-sha", self.identity["reviewed_sha"],
            "--tested-sha", self.identity["tested_sha"],
            "--workflow-sha", self.identity["workflow_sha"],
            "--git-tree-oid", self.identity["git_tree_oid"],
        ]

    def routing(self) -> dict[str, object]:
        return {
            "source": "amd-smi-list-e-json-v1",
            "amd_smi": AMD_SMI_EXECUTABLE,
            "argv": list(AMD_SMI_LIST_COMMAND),
            "bdf": self.row["bdf"],
            "uuid": self.row["uuid"],
            "gpu": 3,
            "hip_id": 17,
        }

    def payload(self, *, dispatch_count: int = 1) -> dict[str, object]:
        return {
            "schema_version": "g1-report-v1",
            "state": "PASS",
            "selected_backend": "hip",
            "fallback_used": False,
            "case_count": len(EXPECTED_SIZES),
            "allocation_count": 12 if dispatch_count else 0,
            "copy_count": 12 if dispatch_count else 0,
            "kernel_dispatch_count": 6 if dispatch_count else 0,
            "dispatch_count": 6 if dispatch_count else 0,
            "cases": [
                {
                    "size": size,
                    "state": "PASS",
                    "byte_exact": True,
                    "dispatch_count": 1,
                    "allocation_count": 2,
                    "copy_count": 2,
                    "timed_out": False,
                    "fallback_used": False,
                }
                for size in EXPECTED_SIZES
            ],
        }

    def execution(self, *, state: str = "PASS", returncode: int = 0, payload: dict[str, object] | None = None) -> runner.EvidenceExecution:
        body = payload if payload is not None else self.payload()
        return runner.EvidenceExecution(
            state=state,
            exit_code=returncode,
            timed_out=state == "TIMEOUT",
            crashed=state == "CRASH",
            stdout=json.dumps(body, sort_keys=True).encode() + b"\n",
            stderr=b"",
            duration_seconds=0.25,
            payload=body,
            error=None if state == "PASS" else f"mocked {state}",
            runtime_binding=self.runtime_binding() if state == "PASS" else None,
            artifact_sha256=sha256_file(self.artifact),
        )

    def close(self) -> None:
        shutil.rmtree(self.stage, ignore_errors=True)
        shutil.rmtree(self.output_root, ignore_errors=True)


class G1RunnerTests(unittest.TestCase):
    def call_main(self, fixture: G1RunnerFixture, *, execution: runner.EvidenceExecution | None = None, extra_env: dict[str, str] | None = None, routing: dict[str, object] | None = None, health_values: tuple[dict[str, object], dict[str, object]] | None = None, process_values: tuple[dict[str, object], dict[str, object]] | None = None, lock=unlocked_host_lock) -> int:
        row = fixture.row
        pre_health, post_health = health_values or (
            health(row, "2026-08-03T08:00:01Z"),
            health(row, "2026-08-03T08:00:06Z"),
        )
        pre_process, post_process = process_values or (
            processes(row, "2026-08-03T08:00:02Z"),
            processes(row, "2026-08-03T08:00:07Z"),
        )
        candidate = {
            "reviewed_sha": fixture.identity["reviewed_sha"],
            "tested_sha": fixture.identity["tested_sha"],
            "workflow_sha": fixture.identity["workflow_sha"],
            "git_tree_oid": fixture.identity["git_tree_oid"],
            "worktree_clean": True,
            "revision_input": "full-sha",
        }
        created = datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc)
        start = datetime(2026, 8, 3, 8, 0, 3, tzinfo=timezone.utc)
        finish = datetime(2026, 8, 3, 8, 0, 5, tzinfo=timezone.utc)
        finished = datetime(2026, 8, 3, 8, 0, 7, tzinfo=timezone.utc)
        with patch.dict("run_g1_evidence.os.environ", extra_env or {}, clear=True), patch(
            "run_g1_evidence.now", side_effect=[created, start, finish, finished]
        ), patch("run_g1_evidence.nonblocking_host_lock", side_effect=lock), patch(
            "run_g1_evidence.git_candidate", return_value=candidate
        ), patch("run_g1_evidence.amd_smi_list_json", return_value=routing or fixture.routing()), patch(
            "run_g1_evidence.observe_health", side_effect=[pre_health, post_health]
        ), patch("run_g1_evidence.observe_processes", side_effect=[pre_process, post_process]), patch(
            "run_g1_evidence.validate_artifact_metadata",
            side_effect=lambda metadata, artifact_path, metadata_path, expected, identity, repo: validate_artifact_metadata(
                metadata, artifact_path, metadata_path, expected, identity, repo, tool_runner=fixture.tool_runner
            ),
        ), patch("run_g1_evidence.run_evidence_binary", return_value=execution or fixture.execution()
        ) as binary_mock:
            result = runner.main(fixture.argv())
        self.binary_mock = binary_mock
        return result

    def test_positive_row_executes_staged_binary_and_writes_strict_outputs(self) -> None:
        fixture = G1RunnerFixture()
        try:
            self.assertEqual(self.call_main(fixture), 0)
            output = fixture.output()
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "report.json", "report.json.sha256", METADATA_NAME,
                    METADATA_NAME + ".sha256", BINARY_NAME, BINARY_NAME + ".sha256",
                },
            )
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            validate_schema(report, json.loads((ROOT / "ci/schema/g1-report-v1.schema.json").read_text()), "G1 report")
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(report["cases"], [
                {
                    "size": size, "state": "PASS", "byte_exact": True,
                    "allocation_count": 2, "copy_count": 2,
                    "kernel_dispatch_count": 1, "dispatch_count": 1,
                    "timed_out": False, "fallback_used": False,
                }
                for size in EXPECTED_SIZES
            ])
            self.assertEqual(report["scope"]["selected_backend"], "hip")
            self.assertFalse(report["scope"]["fallback_used"])
            self.assertEqual(report["scope"]["dispatch_count"], 6)
            self.assertIn("matrix_manifest_sha256", report["artifact"])
            sidecar = (output / "report.json.sha256").read_text(encoding="ascii")
            self.assertEqual(sidecar, f"{sha256_file(output / 'report.json')}  report.json\n")
            self.binary_mock.assert_called_once_with(
                fixture.output() / BINARY_NAME,
                timeout_seconds=300,
                hip_visible_devices="17",
                expected_artifact_sha256=sha256_file(fixture.artifact),
            )
        finally:
            fixture.close()

    def test_identity_mismatch_and_dirty_candidate_fail_before_routing(self) -> None:
        for message in ("dirty worktree", "wrong commit", "wrong tree"):
            fixture = G1RunnerFixture()
            try:
                if message == "wrong tree":
                    candidate = {
                        "reviewed_sha": "a" * 40, "tested_sha": "a" * 40,
                        "workflow_sha": "a" * 40, "git_tree_oid": "c" * 40,
                        "worktree_clean": True, "revision_input": "full-sha",
                    }
                    patcher = patch("run_g1_evidence.git_candidate", return_value=candidate)
                else:
                    patcher = patch("run_g1_evidence.git_candidate", side_effect=ContractError(message))
                with patcher, patch("run_g1_evidence.amd_smi_list_json") as routing_mock, patch.dict("run_g1_evidence.os.environ", {}, clear=True):
                    result = runner.main(fixture.argv())
                self.assertEqual(result, 2)
                routing_mock.assert_not_called()
                self.assertFalse((fixture.output() / "report.json").exists())
            finally:
                fixture.close()

    def test_inherited_visibility_is_rejected_before_canonical_routing(self) -> None:
        fixture = G1RunnerFixture()
        try:
            with patch.dict("run_g1_evidence.os.environ", {"ROCR_VISIBLE_DEVICES": "0"}, clear=True), patch("run_g1_evidence.amd_smi_list_json") as routing_mock:
                result = runner.main(fixture.argv())
            self.assertEqual(result, 2)
            routing_mock.assert_not_called()
            self.assertFalse((fixture.output() / "report.json").exists())
        finally:
            fixture.close()

    def test_wrong_row_gpu_target_toolchain_and_artifact_scope_fail_closed(self) -> None:
        mutations = (
            ("gpu", lambda document: document["gpu"].update({"uuid": "GPU-aaaaaaaaaaaaaaaa"})),
            ("target", lambda document: document.update(target="gfx1201")),
            ("toolchain", lambda document: document.update(toolchain_id="rocm-7.13.0")),
            ("fallback", lambda document: document["scope"].update(cpu_fallback_allowed=True)),
            ("h3", lambda document: document["artifact"].update(path="/tmp/h3/device-code-object-gfx1030.elf")),
        )
        for label, mutation in mutations:
            fixture = G1RunnerFixture()
            try:
                mutation(fixture.metadata)
                fixture.rewrite_metadata()
                with patch.dict("run_g1_evidence.os.environ", {}, clear=True), patch("run_g1_evidence.amd_smi_list_json") as routing_mock:
                    result = runner.main(fixture.argv())
                self.assertEqual(result, 2, label)
                routing_mock.assert_not_called()
                self.assertFalse((fixture.output() / "report.json").exists())
            finally:
                fixture.close()

    def test_sidecars_and_output_paths_are_strict(self) -> None:
        fixture = G1RunnerFixture()
        try:
            fixture.metadata_path.with_name(METADATA_NAME + ".sha256").write_text("0" * 64 + "  wrong\n", encoding="ascii")
            with patch.dict("run_g1_evidence.os.environ", {}, clear=True), patch("run_g1_evidence.amd_smi_list_json") as routing_mock:
                result = runner.main(fixture.argv())
            self.assertEqual(result, 2)
            routing_mock.assert_not_called()
            unsafe_root = Path(tempfile.mkdtemp(prefix="ullm-g1-unsafe-"))
            try:
                symlink_output = unsafe_root / fixture.row_id
                symlink_output.symlink_to(fixture.output_root, target_is_directory=True)
                args = fixture.argv(output=symlink_output)
                with patch.dict("run_g1_evidence.os.environ", {}, clear=True):
                    self.assertEqual(runner.main(args), 2)
            finally:
                shutil.rmtree(unsafe_root, ignore_errors=True)
        finally:
            fixture.close()

    def test_busy_lock_and_pre_observation_failure_do_not_start_binary(self) -> None:
        fixture = G1RunnerFixture()
        try:
            @contextmanager
            def busy_lock(_path: Path):
                raise ContractError("G1 host lock is busy")
                yield

            result = self.call_main(fixture, lock=busy_lock)
            self.assertEqual(result, 2)
            self.binary_mock.assert_not_called()
            fixture.close()
            fixture = G1RunnerFixture()
            unavailable = health(fixture.row, "2026-08-03T08:00:01Z")
            unavailable.update(available=False, reliable=False, source=None)
            result = self.call_main(fixture, health_values=(unavailable, health(fixture.row, "2026-08-03T08:00:03Z")))
            self.assertEqual(result, 2)
            self.binary_mock.assert_not_called()
        finally:
            fixture.close()

    def test_wrong_routing_health_change_and_residual_process_fail_closed(self) -> None:
        cases = []
        fixture = G1RunnerFixture()
        try:
            wrong_routing = fixture.routing()
            wrong_routing["uuid"] = "GPU-aaaaaaaaaaaaaaaa"
            cases.append((wrong_routing, None, None))
            changed = health(fixture.row, "2026-08-03T08:00:03Z")
            changed["facts"]["ras_uncorrectable_count"] = 18
            cases.append((None, (health(fixture.row, "2026-08-03T08:00:01Z"), changed), None))
            residual = processes(fixture.row, "2026-08-03T08:00:04Z")
            residual["residual_runner_children"] = [257]
            cases.append((None, None, (processes(fixture.row, "2026-08-03T08:00:02Z"), residual)))
            for wrong, health_values, process_values in cases:
                result = self.call_main(fixture, routing=wrong, health_values=health_values, process_values=process_values)
                self.assertEqual(result, 2)
                if wrong is not None:
                    self.binary_mock.assert_not_called()
                else:
                    self.binary_mock.assert_called_once()
                shutil.rmtree(fixture.output(), ignore_errors=True)
        finally:
            fixture.close()

    def test_unavailable_timeout_crash_zero_selection_and_fallback_never_pass(self) -> None:
        executions = (
            self.execution_for("UNAVAILABLE", 1, {"schema_version": "g1-report-v1", "state": "UNAVAILABLE", "reason": "HIP unavailable"}),
            self.execution_for("TIMEOUT", None, None),
            self.execution_for("CRASH", -6, None),
        )
        for execution in executions:
            fixture = G1RunnerFixture()
            try:
                result = self.call_main(fixture, execution=execution)
                self.assertEqual(result, 2)
                self.assertFalse((fixture.output() / "report.json").exists())
            finally:
                fixture.close()

        fixture = G1RunnerFixture()
        try:
            zero_payload = fixture.payload(dispatch_count=0)
            zero_payload["cases"] = []
            execution = fixture.execution(state="PASS", payload=zero_payload)
            result = self.call_main(fixture, execution=runner.EvidenceExecution(
                "INFRA_ERROR", 0, False, False, execution.stdout, b"", 0.1, zero_payload, "zero dispatch selection"
            ))
            self.assertEqual(result, 2)
            self.assertFalse((fixture.output() / "report.json").exists())
        finally:
            fixture.close()

    def test_direct_binary_helper_uses_argv_timeout_and_no_cpu_fallback(self) -> None:
        fixture = G1RunnerFixture()
        try:
            payload = json.dumps(fixture.payload(), separators=(",", ":"))
            script = f"#!/bin/sh\n[ \"$1\" = \"--timeout-ms\" ] || exit 9\nprintf '%s\\n' '{payload}'\n"
            fixture.artifact.write_text(script, encoding="utf-8")
            fixture.artifact.chmod(0o700)
            maps = (
                "7f0000000000-7f0000001000 r--p 00000000 00:00 0 /opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000\n"
                "7f0000100000-7f0000101000 r--p 00000000 00:00 0 /opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0\n"
            )
            with patch.object(runner, "_read_process_maps", return_value=maps), patch.object(
                runner, "_read_rocm_release", return_value="7.14.0"
            ):
                result = runner.run_evidence_binary(fixture.artifact, timeout_seconds=2, hip_visible_devices="17")
            self.assertEqual(result.state, "PASS")
            self.assertEqual(result.runtime_binding["rocm_release"], "7.14.0")
            timeout_script = "#!/bin/sh\nsleep 2\n"
            fixture.artifact.write_text(timeout_script, encoding="utf-8")
            result = runner.run_evidence_binary(fixture.artifact, timeout_seconds=1, hip_visible_devices="17")
            self.assertEqual(result.state, "TIMEOUT")
        finally:
            fixture.close()

    def test_descriptor_copy_is_exclusive_and_rejects_symlink_sources(self) -> None:
        fixture = G1RunnerFixture()
        try:
            destination = fixture.output_root / "exclusive-artifact"
            runner._copy_regular(fixture.artifact, destination, "test artifact")
            with self.assertRaises(ContractError):
                runner._copy_regular(fixture.artifact, destination, "test artifact")
            symlink = fixture.output_root / "symlink-artifact"
            symlink.symlink_to(fixture.artifact)
            with self.assertRaises(ContractError):
                runner._snapshot_regular(symlink, "test symlink")
        finally:
            fixture.close()

    def test_loader_binding_missing_wrong_and_duplicate_paths_fail_closed(self) -> None:
        good = {
            "libamdhip64.so.7": {"/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000"},
            "libhsa-runtime64.so.1": {"/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0"},
        }
        environment = {
            "HIP_VISIBLE_DEVICES": "17",
            "PATH": runner.PINNED_PATH,
            "LD_LIBRARY_PATH": runner.PINNED_LD_LIBRARY_PATH,
        }
        with patch.object(runner, "_read_rocm_release", return_value="7.14.0"):
            binding = runner._runtime_binding(good, environment)
            runner._validate_runtime_binding(binding)
            missing = {name: set(paths) for name, paths in good.items()}
            missing["libhsa-runtime64.so.1"].clear()
            with self.assertRaises(ContractError):
                runner._runtime_binding(missing, environment)
            wrong = {name: set(paths) for name, paths in good.items()}
            wrong["libamdhip64.so.7"] = {"/usr/lib/x86_64-linux-gnu/libamdhip64.so.7.0"}
            with self.assertRaises(ContractError):
                runner._runtime_binding(wrong, environment)
            duplicate = {name: set(paths) for name, paths in good.items()}
            duplicate["libamdhip64.so.7"].add("/opt/rocm/other/lib/libamdhip64.so.7.14.60850-0000000")
            with self.assertRaises(ContractError):
                runner._runtime_binding(duplicate, environment)

    def test_timeout_terminates_descendant_process_group(self) -> None:
        fixture = G1RunnerFixture()
        try:
            fixture.artifact.write_text(
                "#!/bin/sh\ntrap '' TERM\n(sleep 30) &\nwait\n",
                encoding="utf-8",
            )
            fixture.artifact.chmod(0o700)
            started = time.monotonic()
            result = runner.run_evidence_binary(fixture.artifact, timeout_seconds=1, hip_visible_devices="17")
            elapsed = time.monotonic() - started
            self.assertEqual(result.state, "TIMEOUT")
            self.assertTrue(result.cleanup_proven)
            self.assertLess(elapsed, 7.0)
        finally:
            fixture.close()

    def test_capture_overflow_terminates_and_reaps_without_oversized_buffers(self) -> None:
        fixture = G1RunnerFixture()
        try:
            fixture.artifact.write_text(
                "#!/bin/sh\nhead -c 1048577 /dev/zero\n",
                encoding="utf-8",
            )
            fixture.artifact.chmod(0o700)
            result = runner.run_evidence_binary(fixture.artifact, timeout_seconds=5, hip_visible_devices="17")
            self.assertEqual(result.state, "INFRA_ERROR")
            self.assertTrue(result.cleanup_proven)
            self.assertLessEqual(len(result.stdout), runner.MAX_CAPTURED_OUTPUT)
            self.assertLessEqual(len(result.stderr), runner.MAX_CAPTURED_OUTPUT)
        finally:
            fixture.close()

    @staticmethod
    def execution_for(state: str, returncode: int | None, payload: dict[str, object] | None) -> runner.EvidenceExecution:
        return runner.EvidenceExecution(
            state=state,
            exit_code=returncode,
            timed_out=state == "TIMEOUT",
            crashed=state == "CRASH",
            stdout=(json.dumps(payload).encode() + b"\n") if payload is not None else b"",
            stderr=b"",
            duration_seconds=0.1,
            payload=payload,
            error=f"mocked {state}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
