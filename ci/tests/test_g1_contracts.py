#!/usr/bin/env python3
"""H0 host-only positive and negative tests for G1 evidence contracts.

These tests validate static contracts and deterministic fixtures only.  They
never execute a GPU binary and their PASS cannot be used as G1 GPU evidence.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import validate_g1_contracts as g1_contracts  # noqa: E402
from aggregate_g1_results import aggregate_results, load_needs, write_summary  # noqa: E402
from common import ContractError, canonical_bytes, sha256_file, sha256_json  # noqa: E402
from validate_g1_contracts import (  # noqa: E402
    BINARY_NAME,
    EXPECTED_ROWS,
    EXPECTED_LOADED_LIBRARIES,
    EXPECTED_LOADER_CONTRACT,
    EXPECTED_SIZES,
    METADATA_NAME,
    inspect_g1_runtime_artifact,
    validate_artifact_metadata,
    validate_g1_matrix,
    validate_report,
    validate_row,
    _manifest_hashes,
)


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return sha256_file(sidecar)


def rebind_staged_paths(row_dir: Path) -> None:
    """Model artifact download staging without changing the source path."""

    report_path = row_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact"]["metadata_path"] = str((row_dir / METADATA_NAME).resolve())
    report["artifact"]["staged_artifact_path"] = str((row_dir / BINARY_NAME).resolve())
    report_path.write_bytes(canonical_bytes(report))
    write_sidecar(report_path)


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
    """Provide deterministic pinned-tool output while exercising the parser."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv, **_kwargs):
        command = tuple(str(item) for item in argv)
        self.commands.append(command)
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-readobj":
            output = device_readobj_fixture(self.target) if command[-1].endswith("device-code-object.elf") else host_readobj_fixture()
            return type("Result", (), {"returncode": 0, "stdout": output.encode(), "stderr": b""})()
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-objcopy":
            destination = next(item.split("=", 2)[2] for item in command if item.startswith("--dump-section=.hip_fatbin="))
            source = Path(command[-2])
            output = Path(command[-1])
            if source.resolve() == output.resolve():
                raise AssertionError("llvm-objcopy source and explicit output must differ")
            output.write_bytes(source.read_bytes())
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


class AggregateFixtureToolRunner:
    """Dispatch deterministic tool fixtures by the staged row being inspected."""

    def __init__(self) -> None:
        self.target = "gfx1030"

    def __call__(self, argv, **kwargs):
        command = tuple(str(item) for item in argv)
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-readobj" and not command[-1].endswith("device-code-object.elf"):
            self.target = "gfx1201" if "g1-gfx1201" in command[-1] else "gfx1030"
        return FixtureToolRunner(self.target)(argv, **kwargs)


class G1Fixture:
    def __init__(self, target: str, run_id: str = "unit-run", attempt: int = 1) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sllm-g1-fixture-"))
        self.row_id = f"g1-{target}"
        self.target = target
        self.row_dir = self.root / self.row_id
        self.row_dir.mkdir()
        matrix = validate_g1_matrix(ROOT)
        self.row = next(row for row in matrix["rows"] if row["row_id"] == self.row_id)
        self.matrix = matrix
        self.identity = {
            "run_id": run_id,
            "run_attempt": attempt,
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40,
        }
        self.artifact = self.row_dir / BINARY_NAME
        artifact_bytes = bytes(((index + (0 if target == "gfx1030" else 1)) % 251 for index in range(257)))
        self.artifact.write_bytes(artifact_bytes)
        self.artifact.chmod(0o700)
        self.source_artifact = self.root / "target" / "release" / BINARY_NAME
        self.source_artifact.parent.mkdir(parents=True)
        self.source_artifact.write_bytes(artifact_bytes)
        self.source_artifact.chmod(0o700)
        self.tool_runner = FixtureToolRunner(target)
        inspection = inspect_g1_runtime_artifact(self.artifact, target, tool_runner=self.tool_runner)
        artifact_record_path = str(self.source_artifact)
        source_artifact_sidecar_sha = write_sidecar(self.source_artifact)
        artifact_sidecar_sha = write_sidecar(self.artifact)
        manifest_hashes = _manifest_hashes(ROOT)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        created_at = now - timedelta(seconds=3)
        started_at = now - timedelta(seconds=2)
        finished_at = now - timedelta(seconds=1)
        timestamp = lambda value: value.isoformat().replace("+00:00", "Z")
        device = {"bdf": self.row["bdf"], "uuid": self.row["uuid"], "target": target}
        health = lambda observed_at: {
            "available": True, "reliable": True, "source": "g0-read-only-health-v1",
            "observed_at": timestamp(observed_at), "device": device, "state": "OK",
            "device_state": "active", "runtime_status": "active", "amdgpu_driver_bound": True,
            "ras_uncorrectable_count": 0, "temperature_c": 42.0,
        }
        process = lambda observed_at: {
            "available": True, "reliable": True, "source": "g0-read-only-process-v1",
            "observed_at": timestamp(observed_at), "device": device, "state": "CLEAN",
            "gpu_processes": [], "residual_runner_children": [],
        }
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
            "gpu": {"bdf": self.row["bdf"], "uuid": self.row["uuid"], "target": target},
            "artifact": {
                "path": artifact_record_path,
                "size_bytes": self.artifact.stat().st_size,
                "sha256": sha256_file(self.source_artifact),
                "sidecar_sha256": source_artifact_sidecar_sha,
                "kind": "dedicated-rust-evidence-binary",
            },
            "observed": inspection["observed"],
            "device_code_sha256": inspection["device_code_sha256"],
            "scope": {
                "model_used": False,
                "cpu_fallback_allowed": False,
                "cpu_fallback_used": False,
                "binary_command": ["target/release/sllm-hip-evidence", "--timeout-ms", "1000"],
            },
        }
        self.metadata_path = self.row_dir / METADATA_NAME
        self.metadata_path.write_bytes(canonical_bytes(self.metadata))
        metadata_sidecar_sha = write_sidecar(self.metadata_path)
        command = ["target/release/sllm-hip-evidence", "--timeout-ms", "1000"]
        self.report = {
            "schema_version": "g1-report-v1",
            "report_id": f"{self.row_id}.{run_id}.{attempt}",
            "row_id": self.row_id,
            "target": target,
            "state": "PASS",
            "required": True,
            "run_id": run_id,
            "run_attempt": attempt,
            "candidate": {
                "reviewed_sha": self.identity["reviewed_sha"],
                "tested_sha": self.identity["tested_sha"],
                "workflow_sha": self.identity["workflow_sha"],
                "git_tree_oid": self.identity["git_tree_oid"],
                "worktree_clean": True,
                "revision_input": "full-sha",
            },
            "artifact": {
                "metadata_path": str(self.metadata_path),
                "metadata_sha256": sha256_file(self.metadata_path),
                "metadata_sidecar_sha256": metadata_sidecar_sha,
                "artifact_path": artifact_record_path,
                "staged_artifact_path": str(self.artifact),
                "artifact_sha256": sha256_file(self.artifact),
                "artifact_sidecar_sha256": artifact_sidecar_sha,
                "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
                "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
                "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
                "target": target,
                "row_id": self.row_id,
                "h3_executable_used": False,
            },
            "execution": {
                "command": command,
                "command_sha256": sha256_json(command),
                "exit_code": 0,
                "timed_out": False,
                "crashed": False,
                "stdout_sha256": "1" * 64,
                "stderr_sha256": "2" * 64,
                "duration_seconds": 1,
            },
            "runtime_binding": {
                **EXPECTED_LOADER_CONTRACT,
                "loaded_libraries": dict(EXPECTED_LOADED_LIBRARIES),
            },
            "scope": {
                "selected_backend": "hip",
                "fallback_allowed": False,
                "fallback_used": False,
                "model_used": False,
                "semantic_op_used": False,
                "byte_exact_verified": True,
                "semantic_numerics_verified": False,
                "allocation_count": 12,
                "copy_count": 12,
                "kernel_dispatch_count": 6,
                "dispatch_count": 6,
            },
            "device": {"bdf": self.row["bdf"], "uuid": self.row["uuid"], "target": target},
            "created_at": timestamp(created_at),
            "started_at": timestamp(started_at),
            "finished_at": timestamp(finished_at),
            "duration_seconds": 1.0,
            "health_pre": health(created_at),
            "health_post": health(now),
            "process_pre": process(created_at),
            "process_post": process(now),
            "cases": [
                {
                    "size": size, "state": "PASS", "byte_exact": True,
                    "allocation_count": 2, "copy_count": 2,
                    "kernel_dispatch_count": 1, "dispatch_count": 1,
                    "timed_out": False, "fallback_used": False,
                }
                for size in EXPECTED_SIZES
            ],
            "error": None,
        }
        self.report_path = self.row_dir / "report.json"
        self.report_path.write_bytes(canonical_bytes(self.report))
        write_sidecar(self.report_path)

    def rewrite(self) -> None:
        self.metadata_path.write_bytes(canonical_bytes(self.metadata))
        write_sidecar(self.metadata_path)
        self.report_path.write_bytes(canonical_bytes(self.report))
        write_sidecar(self.report_path)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class G1ContractTests(unittest.TestCase):
    def test_h0_bounded_inspector_rejects_output_overflow_without_communicate(self) -> None:
        command = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 33)"]
        with self.assertRaises(ContractError):
            g1_contracts.run_bounded_argv(
                command,
                timeout=5.0,
                max_stdout_bytes=32,
                max_stderr_bytes=32,
            )
        self.assertNotIn("communicate(", Path(g1_contracts.__file__).read_text(encoding="utf-8"))

    def test_matrix_is_closed_and_canonical(self) -> None:
        matrix = validate_g1_matrix(ROOT)
        self.assertEqual([row["row_id"] for row in matrix["rows"]], list(EXPECTED_ROWS))
        self.assertEqual(matrix["scope"]["required_sizes"], list(EXPECTED_SIZES))
        self.assertFalse(matrix["execution"]["binary_is_h3_artifact"])

    def test_objcopy_uses_distinct_explicit_output_artifact(self) -> None:
        fixture = G1Fixture("gfx1030")
        try:
            objcopy_calls = [
                command
                for command in fixture.tool_runner.commands
                if command[0] == "/opt/rocm/lib/llvm/bin/llvm-objcopy"
            ]
            self.assertEqual(len(objcopy_calls), 1)
            command = objcopy_calls[0]
            dump_output = next(
                item.split("=", 2)[2]
                for item in command
                if item.startswith("--dump-section=.hip_fatbin=")
            )
            self.assertEqual(Path(command[-2]).resolve(), fixture.artifact.resolve())
            self.assertNotEqual(Path(command[-1]).resolve(), fixture.artifact.resolve())
            self.assertNotEqual(Path(command[-1]).resolve(), Path(dump_output).resolve())
            self.assertIn("sllm-g1-inspect-", command[-1])
        finally:
            fixture.close()

    def test_inspection_detects_artifact_mutation_and_fails_closed(self) -> None:
        class MutatingToolRunner(FixtureToolRunner):
            def __call__(self, argv, **kwargs):
                command = tuple(str(item) for item in argv)
                result = super().__call__(command, **kwargs)
                if command[0] == "/opt/rocm/lib/llvm/bin/llvm-objcopy":
                    Path(command[-2]).write_bytes(b"mutated-by-inspector")
                return result

        fixture = G1Fixture("gfx1030")
        try:
            before = fixture.artifact.read_bytes()
            with self.assertRaisesRegex(ContractError, "changed during inspection"):
                inspect_g1_runtime_artifact(
                    fixture.artifact,
                    fixture.target,
                    tool_runner=MutatingToolRunner(fixture.target),
                )
            self.assertNotEqual(fixture.artifact.read_bytes(), before)
        finally:
            fixture.close()

    def test_wrong_identity_and_target_fail(self) -> None:
        fixture = G1Fixture("gfx1030")
        try:
            bad = copy.deepcopy(fixture.metadata)
            bad["gpu"]["uuid"] = "GPU-aaaaaaaaaaaaaaaa"
            with self.assertRaises(ContractError):
                validate_artifact_metadata(bad, fixture.artifact, fixture.metadata_path, fixture.row, fixture.identity, tool_runner=fixture.tool_runner)
            bad = copy.deepcopy(fixture.metadata)
            bad["candidate"]["tested_sha"] = "c" * 40
            with self.assertRaises(ContractError):
                validate_artifact_metadata(bad, fixture.artifact, fixture.metadata_path, fixture.row, tool_runner=fixture.tool_runner)
            bad = copy.deepcopy(fixture.metadata)
            bad["candidate"]["worktree_clean"] = False
            with self.assertRaises(ContractError):
                validate_artifact_metadata(bad, fixture.artifact, fixture.metadata_path, fixture.row, tool_runner=fixture.tool_runner)
            bad = copy.deepcopy(fixture.report)
            bad["target"] = "gfx1201"
            fixture.report = bad
            fixture.rewrite()
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
        finally:
            fixture.close()

    def test_h3_substitution_and_scope_are_rejected(self) -> None:
        fixture = G1Fixture("gfx1030")
        try:
            bad = copy.deepcopy(fixture.metadata)
            bad["artifact"]["path"] = "/tmp/h3-gfx1030/device-code-object-gfx1030.elf"
            with self.assertRaises(ContractError):
                validate_artifact_metadata(bad, fixture.artifact, fixture.metadata_path, fixture.row, fixture.identity, tool_runner=fixture.tool_runner)
            bad = copy.deepcopy(fixture.metadata)
            bad["scope"]["binary_command"] = ["target/release/h3-compile-probe"]
            with self.assertRaises(ContractError):
                validate_artifact_metadata(bad, fixture.artifact, fixture.metadata_path, fixture.row, fixture.identity, tool_runner=fixture.tool_runner)
        finally:
            fixture.close()

    def test_stale_hashes_and_no_dispatch_fail(self) -> None:
        fixture = G1Fixture("gfx1030")
        try:
            fixture.artifact.write_bytes(b"tampered")
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
        finally:
            fixture.close()
        fixture = G1Fixture("gfx1030")
        try:
            fixture.report["scope"]["dispatch_count"] = 0
            fixture.rewrite()
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
        finally:
            fixture.close()

    def test_observed_code_object_tuple_and_digest_are_recomputed(self) -> None:
        fixture = G1Fixture("gfx1030")
        try:
            bad = copy.deepcopy(fixture.metadata)
            bad["device_code_sha256"] = "0" * 64
            with self.assertRaises(ContractError):
                validate_artifact_metadata(
                    bad,
                    fixture.artifact,
                    fixture.metadata_path,
                    fixture.row,
                    fixture.identity,
                    tool_runner=fixture.tool_runner,
                )
            bad = copy.deepcopy(fixture.metadata)
            bad["observed"]["e_flags"] = "0x0000004e"
            with self.assertRaises(ContractError):
                validate_artifact_metadata(
                    bad,
                    fixture.artifact,
                    fixture.metadata_path,
                    fixture.row,
                    fixture.identity,
                    tool_runner=fixture.tool_runner,
                )
        finally:
            fixture.close()

    def test_runtime_binding_is_mandatory_canonical_and_closed(self) -> None:
        mutations = (
            ("missing binding", lambda report: report.pop("runtime_binding")),
            ("extra binding key", lambda report: report["runtime_binding"].update({"extra": True})),
            ("wrong root", lambda report: report["runtime_binding"].update({"rocm_root": "/usr/local/rocm"})),
            ("wrong release", lambda report: report["runtime_binding"].update({"rocm_release": "7.13.0"})),
            ("wrong PATH", lambda report: report["runtime_binding"].update({"path": "/opt/rocm/bin"})),
            ("wrong LD_LIBRARY_PATH", lambda report: report["runtime_binding"].update({"ld_library_path": "/opt/rocm/lib"})),
            ("wrong observation", lambda report: report["runtime_binding"].update({"observation_method": "maps"})),
            ("wrong required library order", lambda report: report["runtime_binding"].update({"required_libraries": list(reversed(EXPECTED_LOADER_CONTRACT["required_libraries"]))})),
            ("inherited environment", lambda report: report["runtime_binding"].update({"inherited_loader_environment": True})),
            ("missing library", lambda report: report["runtime_binding"]["loaded_libraries"].pop("libhsa-runtime64.so.1")),
            ("extra library", lambda report: report["runtime_binding"]["loaded_libraries"].update({"libamdhip64.so": "/opt/rocm/core-7.14/lib/libamdhip64.so"})),
            ("outside root", lambda report: report["runtime_binding"]["loaded_libraries"].update({"libamdhip64.so.7": "/usr/lib/libamdhip64.so.7.0"})),
            ("wrong soname", lambda report: report["runtime_binding"]["loaded_libraries"].update({"libamdhip64.so.7": "/opt/rocm/core-7.14/lib/libamdhip64.so.8.0"})),
            ("path alias", lambda report: report["runtime_binding"]["loaded_libraries"].update({"libamdhip64.so.7": "/opt/rocm/core-7.14/lib/../lib/libamdhip64.so.7.14.60850-0000000"})),
            ("duplicate path", lambda report: report["runtime_binding"]["loaded_libraries"].update({"libhsa-runtime64.so.1": EXPECTED_LOADED_LIBRARIES["libamdhip64.so.7"]})),
        )
        for label, mutation in mutations:
            fixture = G1Fixture("gfx1030")
            try:
                mutation(fixture.report)
                fixture.rewrite()
                with self.subTest(label=label), self.assertRaises(ContractError):
                    validate_row(
                        fixture.row_dir,
                        fixture.row_id,
                        fixture.row,
                        fixture.identity,
                        fixture.matrix,
                        tool_runner=fixture.tool_runner,
                    )
            finally:
                fixture.close()

    def test_nonpass_states_and_fallback_fail(self) -> None:
        for field, value in (("state", "TIMEOUT"), ("state", "UNAVAILABLE"), ("state", "CRASH"), ("state", "SKIP")):
            fixture = G1Fixture("gfx1030")
            try:
                fixture.report[field] = value
                fixture.rewrite()
                with self.assertRaises(ContractError):
                    validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
            finally:
                fixture.close()
        fixture = G1Fixture("gfx1030")
        try:
            fixture.report["scope"]["fallback_used"] = True
            fixture.rewrite()
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
        finally:
            fixture.close()

    def test_timestamps_health_counts_and_source_staging_binding_fail(self) -> None:
        mutations = (
            ("future finish", lambda report: report.update({"finished_at": "2999-01-01T00:00:00Z"})),
            ("unavailable health", lambda report: report["health_post"].update({"available": False})),
            ("residual process", lambda report: report["process_post"].update({"residual_runner_children": [{"pid": 1}]})),
            ("allocation total", lambda report: report["scope"].update({"allocation_count": 11})),
            ("case copy total", lambda report: report["cases"][2].update({"copy_count": 1})),
            ("source path mismatch", lambda report: report["artifact"].update({"artifact_path": "/tmp/target/release/sllm-hip-evidence"})),
            ("staged path mismatch", lambda report: report["artifact"].update({"staged_artifact_path": "/tmp/other/sllm-hip-evidence"})),
        )
        for label, mutation in mutations:
            fixture = G1Fixture("gfx1030")
            try:
                mutation(fixture.report)
                fixture.rewrite()
                with self.subTest(label=label), self.assertRaises(ContractError):
                    validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.identity, fixture.matrix, tool_runner=fixture.tool_runner)
            finally:
                fixture.close()

    def test_symlink_and_non_private_aggregate_output_fail(self) -> None:
        fixture = G1Fixture("gfx1030")
        outside_root = Path(tempfile.mkdtemp(prefix="sllm-g1-outside-root-"))
        outside = outside_root / "output"
        private = Path(tempfile.mkdtemp(prefix="sllm-g1-private-"))
        link = Path(tempfile.mkdtemp(prefix="sllm-g1-link-parent-")) / "link"
        try:
            with self.assertRaises(ContractError):
                write_summary(outside, {"not": "an aggregate"})
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaises(ContractError):
                write_summary(link, {"not": "an aggregate"})
            private.chmod(0o755)
            with self.assertRaises(ContractError):
                write_summary(private, {"not": "an aggregate"})
        finally:
            fixture.close()
            shutil.rmtree(outside_root, ignore_errors=True)
            shutil.rmtree(private, ignore_errors=True)
            shutil.rmtree(link.parent, ignore_errors=True)

    def test_dirty_checkout_is_not_a_pass_candidate(self) -> None:
        from unittest.mock import patch

        identity = {
            "run_id": "unit-run", "run_attempt": 1,
            "reviewed_sha": "a" * 40, "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40,
        }
        with patch("aggregate_g1_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), patch(
            "aggregate_g1_results.ensure_clean_worktree", side_effect=ContractError("dirty")
        ), self.assertRaises(ContractError):
            aggregate_results(
                needs_path=Path("/tmp/missing-g1-needs.json"),
                artifact_dir=Path("/tmp/sllm-g1-missing"), repo=ROOT,
                output_dir=Path("/tmp/sllm-g1-summary"), run_id=identity["run_id"],
                run_attempt=identity["run_attempt"], reviewed_sha=identity["reviewed_sha"],
                tested_sha=identity["tested_sha"], workflow_sha=identity["workflow_sha"],
                tree_oid=identity["git_tree_oid"],
            )

    def test_missing_duplicate_and_cross_row_mismatch_fail(self) -> None:
        first = G1Fixture("gfx1030")
        second = G1Fixture("gfx1201")
        collection = Path(tempfile.mkdtemp(prefix="sllm-g1-collection-"))
        needs = collection.parent / "g1-needs.json"
        try:
            shutil.copytree(first.row_dir, collection / first.row_id)
            shutil.copytree(second.row_dir, collection / second.row_id)
            rebind_staged_paths(collection / first.row_id)
            rebind_staged_paths(collection / second.row_id)
            needs.write_text(json.dumps({row_id: {"result": "success"} for row_id in EXPECTED_ROWS}) + "\n", encoding="utf-8")
            identity = first.identity
            from unittest.mock import patch
            aggregate_runner = AggregateFixtureToolRunner()
            with patch("aggregate_g1_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), patch("aggregate_g1_results.ensure_clean_worktree"):
                result = aggregate_results(
                    needs_path=needs, artifact_dir=collection, repo=ROOT, output_dir=collection / "out",
                    run_id=identity["run_id"], run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"], tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"], tree_oid=identity["git_tree_oid"],
                    tool_runner=aggregate_runner,
                )
            self.assertEqual(result["state"], "PASS")
            output = Path(tempfile.mkdtemp(prefix="sllm-g1-summary-"))
            write_summary(output, result, ROOT)
            with self.assertRaises(ContractError):
                write_summary(output, result, ROOT)
            shutil.rmtree(output, ignore_errors=True)
            victim = collection / "victim"
            victim.write_bytes(b"must-not-change")
            symlink_output = Path(tempfile.mkdtemp(prefix="sllm-g1-symlink-output-"))
            (symlink_output / "aggregate.json").symlink_to(victim)
            with self.assertRaises(ContractError):
                write_summary(symlink_output, result, ROOT)
            self.assertEqual(victim.read_bytes(), b"must-not-change")
            shutil.rmtree(symlink_output, ignore_errors=True)
            stale_sidecar_output = Path(tempfile.mkdtemp(prefix="sllm-g1-stale-sidecar-"))
            stale_sidecar = stale_sidecar_output / "aggregate.json.sha256"
            stale_sidecar.write_bytes(b"stale-sidecar")
            with self.assertRaises(ContractError):
                write_summary(stale_sidecar_output, result, ROOT)
            self.assertEqual(stale_sidecar.read_bytes(), b"stale-sidecar")
            shutil.rmtree(stale_sidecar_output, ignore_errors=True)
            second_report_path = collection / "g1-gfx1201" / "report.json"
            second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
            second_report["run_id"] = "different-run"
            second_report_path.write_bytes(canonical_bytes(second_report))
            write_sidecar(second_report_path)
            with patch("aggregate_g1_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), patch("aggregate_g1_results.ensure_clean_worktree"), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs, artifact_dir=collection, repo=ROOT, output_dir=collection / "out",
                    run_id=identity["run_id"], run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"], tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"], tree_oid=identity["git_tree_oid"],
                    tool_runner=aggregate_runner,
                )
            shutil.copytree(second.row_dir, collection / "g1-gfx1201-rebuilt")
            rebind_staged_paths(collection / "g1-gfx1201-rebuilt")
            shutil.rmtree(collection / "g1-gfx1201")
            (collection / "g1-gfx1201-rebuilt").rename(collection / "g1-gfx1201")
            shutil.copytree(collection / first.row_id, collection / "g1-extra")
            with patch("aggregate_g1_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), patch("aggregate_g1_results.ensure_clean_worktree"), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs, artifact_dir=collection, repo=ROOT, output_dir=collection / "out",
                    run_id=identity["run_id"], run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"], tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"], tree_oid=identity["git_tree_oid"],
                    tool_runner=aggregate_runner,
                )
            shutil.rmtree(collection / "g1-extra")
            (collection / "g1-gfx1201" / METADATA_NAME).unlink()
            with patch("aggregate_g1_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), patch("aggregate_g1_results.ensure_clean_worktree"), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs, artifact_dir=collection, repo=ROOT, output_dir=collection / "out",
                    run_id=identity["run_id"], run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"], tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"], tree_oid=identity["git_tree_oid"],
                )
        finally:
            first.close()
            second.close()
            shutil.rmtree(collection, ignore_errors=True)
            needs.unlink(missing_ok=True)

    def test_needs_order_and_cross_row_identity_fail(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="sllm-g1-needs-")) / "needs.json"
        try:
            path.write_text(json.dumps({"g1-gfx1201": {"result": "success"}, "g1-gfx1030": {"result": "success"}}), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_needs(path)
            path.write_text(json.dumps({row_id: {"result": "success"} for row_id in EXPECTED_ROWS} | {"extra": {"result": "success"}}), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_needs(path)
            path.write_text(json.dumps({"g1-gfx1030": {"result": "success"}, "g1-gfx1201": {"result": "skipped"}}), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_needs(path)
            first = G1Fixture("gfx1030")
            try:
                first.report["candidate"]["git_tree_oid"] = "c" * 40
                first.rewrite()
                with self.assertRaises(ContractError):
                    validate_row(first.row_dir, first.row_id, first.row, first.identity, first.matrix, tool_runner=first.tool_runner)
            finally:
                first.close()
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
