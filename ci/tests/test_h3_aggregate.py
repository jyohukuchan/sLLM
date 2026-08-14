#!/usr/bin/env python3
"""Boundary and negative tests for the independent H3 aggregate contract."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from aggregate_h3_results import (  # noqa: E402
    EXPECTED_ENVIRONMENT,
    EXPECTED_ROWS,
    EXPECTED_SCOPE,
    ContractError,
    aggregate_results,
    canonical_bytes,
    load_contract,
    load_needs,
    sha256_file,
    sha256_json,
    validate_row,
)
from common import ContractError as CommonContractError  # noqa: E402
from validate_json_manifests import (  # noqa: E402
    validate_h3_workflow,
    validate_host_workflow,
    workflow_documents,
)


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return sha256_file(sidecar)


class H3Fixture:
    def __init__(self, target: str, size: int = 257) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sllm-h3-aggregate-test-"))
        self.row_id = f"h3-{target}"
        self.target = target
        self.row_dir = self.root / self.row_id
        self.row_dir.mkdir()
        self.toolchain, self.matrix, rows = load_contract(ROOT)
        self.row = rows[self.row_id]
        self.identity = {
            "run_id": "unit-run",
            "run_attempt": 1,
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40,
        }
        self.artifact = self.row_dir / f"device-code-object-{target}.elf"
        self.artifact.write_bytes(bytes((index % 251 for index in range(size))))
        output_directory = f"/tmp/sllm-h3-unit/h3-{target}"
        artifact_path = f"{output_directory}/device-code-object-{target}.elf"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        started = now - timedelta(seconds=1)
        self.timestamps = {
            "created_at": started.isoformat().replace("+00:00", "Z"),
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": now.isoformat().replace("+00:00", "Z"),
        }
        self.metadata = {
            "schema_version": "hip-artifact-metadata-v1",
            "metadata_id": f"h3-artifact-{target}",
            "matrix_row_id": self.row_id,
            "target": target,
            "candidate": {
                "commit_sha": self.identity["reviewed_sha"],
                "tree_oid": self.identity["git_tree_oid"],
                "reviewed_sha": self.identity["reviewed_sha"],
                "tested_sha": self.identity["tested_sha"],
                "workflow_sha": self.identity["workflow_sha"],
            },
            "toolchain_id": "rocm-7.14.0",
            "matrix_id": "hip-compile-v1",
            "toolchain_manifest_sha256": sha256_json(self.toolchain),
            "matrix_manifest_sha256": sha256_json(self.matrix),
            "image": {key: self.toolchain["image"][key] for key in (
                "repository", "tag", "manifest_digest", "config_digest",
                "manifest_list_digest", "manifest_type", "platform",
            )},
            "resolved_paths": copy.deepcopy(self.toolchain["paths"]),
            "build": {
                "source_directory": "/workspace",
                "source_path": "/workspace/native/hip/src/hip_compile_probe.hip.cpp",
                "output_directory": output_directory,
                "object_path": f"{output_directory}/hip-compile-probe-{target}.o",
                "link_output_path": f"{output_directory}/hip-compile-probe-{target}.elf",
                "generator": "direct-amdclang++",
                "mode": "direct-compile-link",
                "build_type": "Release",
                "language_standard": "gnu++17",
                "output_directory_scope": "row-private",
                "source_tree_output": False,
                "shared_build_directory": False,
            },
            "codegen": copy.deepcopy(self.row["codegen"]),
            "artifact": {
                "path": artifact_path,
                "size_bytes": size,
                "sha256": sha256_file(self.artifact),
            },
            "host_bundle": {
                "format": "ELF64",
                "machine": "X86_64",
                "bundles": [
                    {"id": f"hipv4-amdgcn-amd-amdhsa--{target}", "target": target},
                    {"id": "host-x86_64-unknown-linux-gnu-", "target": "host"},
                ],
                "sections": {".hip_fatbin": {"present": True, "size_bytes": 256}},
            },
            "device_code_object": {
                "format": "ELF64",
                "machine": "AMDGPU",
                "target": target,
                "ei_abiversion": 4,
                "e_flags": {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}[target],
                "code_object_version": "V6",
                "wavefront_size": 32,
                "features": {
                    "xnack": "unsupported",
                    "sramecc": "unsupported",
                    "generic_processor_version": 0,
                },
                "sections": {".text": {"present": True, "size_bytes": 255}},
                "symbols": [{"name": "sllm_hip_compile_probe", "defined": True}],
            },
            "scope": EXPECTED_SCOPE.copy(),
            "execution_environment": EXPECTED_ENVIRONMENT.copy(),
            "timestamps": self.timestamps,
            "duration_seconds": 1,
        }
        self.metadata_path = self.row_dir / "hip-artifact-metadata.json"
        self.metadata_path.write_bytes(canonical_bytes(self.metadata))
        metadata_sidecar_sha = write_sidecar(self.metadata_path)
        commands = [
            [
                "/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG",
                "-std=gnu++17", f"--offload-arch={target}", "-mcode-object-version=6",
                "-mno-wavefrontsize64", "-o", f"{output_directory}/hip-compile-probe-{target}.o",
                "-x", "hip", "-c", "/workspace/native/hip/src/hip_compile_probe.hip.cpp",
            ],
            [
                "/opt/rocm/bin/amdclang++", "-O3", "-DNDEBUG", f"--offload-arch={target}",
                "-mcode-object-version=6", "-mno-wavefrontsize64", "--hip-link",
                "--rtlib=compiler-rt", "-unwindlib=libgcc",
                f"{output_directory}/hip-compile-probe-{target}.o", "-o",
                f"{output_directory}/hip-compile-probe-{target}.elf", "/opt/rocm/lib/libamdhip64.so",
            ],
        ]
        h3_toolchain = {
            "toolchain_id": self.toolchain["toolchain_id"],
            "manifest_sha256": sha256_json(self.toolchain),
            "rocm": self.toolchain["rocm"],
            "compiler": self.toolchain["compiler"],
            "paths": self.toolchain["paths"],
            "observed": {
                "compiler_version": "AMD clang version 23.0.0",
                "llvm_major": 23,
                "tools": {name: "LLVM version 23.0.0" for name in (
                    "clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump",
                )},
            },
        }
        artifact_sha = sha256_file(self.artifact)
        self.report = {
            "schema_version": "test-result-v1",
            "result_id": f"{self.row_id}.unit-run.1",
            "suite_id": self.row_id,
            "tier": "tier_h3",
            "state": "PASS",
            "required": False,
            "evidence_mode": "required-ci",
            **self.identity,
            "worktree_clean": True,
            "matrix_manifest_sha256": sha256_json(self.matrix),
            "matrix_row_id": self.row_id,
            "tuple_digest": sha256_json(self.row),
            "command": commands,
            "command_sha256": sha256_json(commands),
            "toolchain": {"h3": h3_toolchain},
            "toolchain_sha256": sha256_json({"h3": h3_toolchain}),
            "artifact": {"content_sha256": artifact_sha, "manifest_sha256": sha256_file(self.metadata_path)},
            "h3_artifact": {
                "target": target,
                "size_bytes": size,
                "content_sha256": artifact_sha,
                "metadata_sha256": sha256_file(self.metadata_path),
                "metadata_sidecar_sha256": metadata_sidecar_sha,
                "artifact_sidecar_sha256": "pending",
            },
            "h3_scope": EXPECTED_SCOPE.copy(),
            "created_at": self.timestamps["created_at"],
            "started_at": self.timestamps["started_at"],
            "finished_at": self.timestamps["finished_at"],
            "duration_seconds": 1,
            "seed": self.row["seed"],
            "counts": {"collected": 2, "selected": 2, "passed": 2, "failed": 0, "skipped": 0, "deselected": 0},
            "steps": [
                {"step_id": f"{self.row_id}.command-1", "state": "PASS", "selection_required": True, "resource": {"network_isolated": True, "network_guard_strategy": "container-network-none"}},
                {"step_id": f"{self.row_id}.command-2", "state": "PASS", "selection_required": True, "resource": {"network_isolated": True, "network_guard_strategy": "container-network-none"}},
            ],
            "diagnostic": {"errors": [], "network_disabled": True, "model_disabled": True, "gpu_fallback_disabled": True, "network_guard_self_test": True},
            "execution_environment": EXPECTED_ENVIRONMENT.copy(),
        }
        artifact_sidecar_sha = write_sidecar(self.artifact)
        self.report["h3_artifact"]["artifact_sidecar_sha256"] = artifact_sidecar_sha
        self.report_path = self.row_dir / "report.json"
        self.report_path.write_bytes(canonical_bytes(self.report))
        write_sidecar(self.report_path)

    def close(self) -> None:
        shutil.rmtree(self.root)


class H3AggregateContractTests(unittest.TestCase):
    def _collection(self) -> tuple[Path, Path, dict[str, str], list[H3Fixture]]:
        fixtures = [H3Fixture(target) for target in ("gfx1030", "gfx1201")]
        collection = Path(tempfile.mkdtemp(prefix="sllm-h3-collection-"))
        for fixture in fixtures:
            shutil.copytree(fixture.row_dir, collection / fixture.row_id)
        needs = collection.parent / f"{collection.name}-needs.json"
        needs.write_text(json.dumps({row_id: {"result": "success"} for row_id in EXPECTED_ROWS}) + "\n", encoding="utf-8")
        return collection, needs, fixtures[0].identity, fixtures

    def test_collection_success_and_missing_duplicate_unknown_rows_fail_closed(self) -> None:
        collection, needs, identity, fixtures = self._collection()
        output = collection.parent / "aggregate"
        try:
            with patch("aggregate_h3_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}):
                summary = aggregate_results(
                    needs_path=needs,
                    artifact_dir=collection,
                    repo=ROOT,
                    output_dir=output,
                    run_id=identity["run_id"],
                    run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"],
                    tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"],
                    tree_oid=identity["git_tree_oid"],
                )
            self.assertEqual(summary["state"], "PASS")
            shutil.copytree(collection / "h3-gfx1030", collection / "h3-gfx1030-copy")
            with patch("aggregate_h3_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs,
                    artifact_dir=collection,
                    repo=ROOT,
                    output_dir=output,
                    run_id=identity["run_id"],
                    run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"],
                    tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"],
                    tree_oid=identity["git_tree_oid"],
                )
            shutil.rmtree(collection / "h3-gfx1030-copy")
            (collection / "h3-gfx1201").rename(collection / "h3-gfx1201-copy")
            with patch("aggregate_h3_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs,
                    artifact_dir=collection,
                    repo=ROOT,
                    output_dir=output,
                    run_id=identity["run_id"],
                    run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"],
                    tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"],
                    tree_oid=identity["git_tree_oid"],
                )
            (collection / "h3-gfx1201-copy").rename(collection / "h3-gfx1201")
            shutil.rmtree(collection / "h3-gfx1030")
            with patch("aggregate_h3_results.git_identity", return_value={"commit": identity["reviewed_sha"], "tree": identity["git_tree_oid"]}), self.assertRaises(ContractError):
                aggregate_results(
                    needs_path=needs,
                    artifact_dir=collection,
                    repo=ROOT,
                    output_dir=output,
                    run_id=identity["run_id"],
                    run_attempt=identity["run_attempt"],
                    reviewed_sha=identity["reviewed_sha"],
                    tested_sha=identity["tested_sha"],
                    workflow_sha=identity["workflow_sha"],
                    tree_oid=identity["git_tree_oid"],
                )
        finally:
            for fixture in fixtures:
                fixture.close()
            shutil.rmtree(collection, ignore_errors=True)
            needs.unlink(missing_ok=True)
            shutil.rmtree(output, ignore_errors=True)

    def test_boundary_artifact_sizes_255_256_257(self) -> None:
        for size in (255, 256, 257):
            fixture = H3Fixture("gfx1030", size)
            try:
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
            finally:
                fixture.close()

    def test_exact_two_rows_and_target_specific_identity(self) -> None:
        fixtures = [H3Fixture(target) for target in ("gfx1030", "gfx1201")]
        try:
            summaries = [validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity) for fixture in fixtures]
            self.assertEqual([summary["target"] for summary in summaries], ["gfx1030", "gfx1201"])
            fixtures[1].metadata["target"] = "gfx1030"
            fixtures[1].metadata_path.write_bytes(canonical_bytes(fixtures[1].metadata))
            with self.assertRaises(ContractError):
                validate_row(fixtures[1].row_dir, fixtures[1].row_id, fixtures[1].row, fixtures[1].toolchain, fixtures[1].matrix, fixtures[1].identity)
        finally:
            for fixture in fixtures:
                fixture.close()

    def test_direct_commands_reject_cmake_missing_extra_and_target_substitution(self) -> None:
        mutations = (
            (lambda commands: commands[0].__setitem__(0, "cmake"), "CMake command"),
            (lambda commands: commands[0].remove("-mno-wavefrontsize64"), "missing direct flag"),
            (lambda commands: commands[1].append("-Winvalid-pch"), "extra direct flag"),
            (lambda commands: commands[1].__setitem__(3, "--offload-arch=gfx1201"), "link target substitution"),
            (lambda commands: commands[1].__setitem__(9, "/tmp/sllm-h3-unit/h3-gfx1201/hip-compile-probe-gfx1201.o"), "object linkage substitution"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = H3Fixture("gfx1030")
                try:
                    mutation(fixture.report["command"])
                    fixture.report["command_sha256"] = sha256_json(fixture.report["command"])
                    fixture.report_path.write_bytes(canonical_bytes(fixture.report))
                    write_sidecar(fixture.report_path)
                    with self.assertRaises(ContractError):
                        validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
                finally:
                    fixture.close()

    def test_sidecar_fail_closed_and_unknown_file_rejected(self) -> None:
        fixture = H3Fixture("gfx1030")
        try:
            fixture.row_dir.joinpath("unexpected.bin").write_bytes(b"unknown")
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
            fixture.row_dir.joinpath("unexpected.bin").unlink()
            fixture.row_dir.joinpath("report.json.sha256").write_text("0" * 64 + "  report.json\n", encoding="ascii")
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
        finally:
            fixture.close()

    def test_tamper_stale_fail_and_infra_error_reports_are_rejected(self) -> None:
        for state in ("FAIL", "INFRA_ERROR"):
            with self.subTest(state=state):
                fixture = H3Fixture("gfx1030")
                try:
                    fixture.report["state"] = state
                    fixture.report_path.write_bytes(canonical_bytes(fixture.report))
                    write_sidecar(fixture.report_path)
                    with self.assertRaises(ContractError):
                        validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
                finally:
                    fixture.close()

        fixture = H3Fixture("gfx1030")
        try:
            fixture.report["reviewed_sha"] = "c" * 40
            fixture.report_path.write_bytes(canonical_bytes(fixture.report))
            write_sidecar(fixture.report_path)
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
        finally:
            fixture.close()

        fixture = H3Fixture("gfx1030")
        try:
            fixture.artifact.write_bytes(b"tampered" * 37)
            write_sidecar(fixture.artifact)
            with self.assertRaises(ContractError):
                validate_row(fixture.row_dir, fixture.row_id, fixture.row, fixture.toolchain, fixture.matrix, fixture.identity)
        finally:
            fixture.close()

    def test_fail_state_and_needs_missing_unknown_non_success(self) -> None:
        needs = Path(tempfile.mkdtemp(prefix="sllm-h3-needs-")) / "needs.json"
        try:
            needs.write_text(json.dumps({"h3-gfx1030": {"result": "success"}}) + "\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_needs(needs)
            for result in ("failure", "cancelled", "skipped", "timed_out"):
                with self.subTest(result=result):
                    needs.write_text(json.dumps({"h3-gfx1030": {"result": "success"}, "h3-gfx1201": {"result": result}}) + "\n", encoding="utf-8")
                    with self.assertRaises(ContractError):
                        load_needs(needs)
            needs.write_text(json.dumps({"h3-gfx1030": {"result": "success"}, "h3-gfx1201": {"result": "success"}, "h3-unknown": {"result": "success"}}) + "\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_needs(needs)
        finally:
            shutil.rmtree(needs.parent)

    def test_h3_profile_and_host_required_separation(self) -> None:
        documents = dict(workflow_documents())
        h3_path = ROOT / ".github/workflows/h3-compile.yml"
        host_path = ROOT / ".github/workflows/host-required.yml"
        validate_h3_workflow(h3_path, documents[h3_path])

        h3 = copy.deepcopy(documents[h3_path])
        h3["jobs"]["h3-aggregate"]["timeout-minutes"] = 2
        with self.assertRaises(CommonContractError):
            validate_h3_workflow(h3_path, h3)
        h3 = copy.deepcopy(documents[h3_path])
        h3["jobs"]["h3-aggregate"]["steps"][0]["with"]["fetch-depth"] = 0
        with self.assertRaises(CommonContractError):
            validate_h3_workflow(h3_path, h3)

        git_mount = "--mount \"type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly\" \\\n"
        for mutation in (
            ("dst=/output", "dst=/output,rw"),
            ("dst=/output", "dst=/output,readonly"),
            ("dst=/workspace,readonly", "dst=/workspace"),
            (git_mount, ""),
            ("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/usr/bin/git,dst=/usr/local/bin/git,rw"),
            ("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/bin/git,dst=/usr/local/bin/git,readonly"),
            ("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/usr/bin/git,dst=/bin/git,readonly"),
            (git_mount, git_mount + "            --mount \"type=bind,src=/usr/bin/python3,dst=/usr/local/bin/python3,readonly\" \\\n"),
        ):
            invalid_h3 = copy.deepcopy(documents[h3_path])
            for row_id in ("h3-gfx1030", "h3-gfx1201"):
                for step in invalid_h3["jobs"][row_id]["steps"]:
                    if isinstance(step, dict) and isinstance(step.get("run"), str):
                        step["run"] = step["run"].replace(*mutation)
            with self.assertRaises(CommonContractError):
                validate_h3_workflow(h3_path, invalid_h3)

        validate_host_workflow(host_path, documents[host_path])
        host = copy.deepcopy(documents[host_path])
        host["jobs"]["h0"]["timeout-minutes"] = 14
        with self.assertRaises(CommonContractError):
            validate_host_workflow(host_path, host)
        host = copy.deepcopy(documents[host_path])
        host["jobs"]["host-required"]["needs"].append("h3-gfx1030")
        with self.assertRaises(CommonContractError):
            validate_host_workflow(host_path, host)
        host = copy.deepcopy(documents[host_path])
        host["jobs"]["h3-gfx1030"] = copy.deepcopy(host["jobs"]["h0"])
        with self.assertRaises(CommonContractError):
            validate_host_workflow(host_path, host)


if __name__ == "__main__":
    unittest.main(verbosity=2)
