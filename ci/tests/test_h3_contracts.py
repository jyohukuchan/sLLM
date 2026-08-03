#!/usr/bin/env python3
"""Negative and boundary tests for the static Phase 2 H3 contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, sha256_json  # noqa: E402
from validate_h3_contracts import (  # noqa: E402
    EXPECTED_TARGETS,
    validate_artifact_metadata,
    validate_h3_contracts,
    validate_h3_manifests,
)


CONTRACT_FILES = (
    "ci/schema/rocm-toolchain-v1.schema.json",
    "ci/schema/hip-artifact-metadata-v1.schema.json",
    "ci/toolchains/rocm-7.14.0.json",
    "ci/matrix/hip-compile-v1.json",
)


def copy_contract_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="ullm-h3-contract-repo-"))
    for relative in CONTRACT_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return root


def mutate_json(root: Path, relative: str, mutation) -> None:
    path = root / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def artifact_fixture(
    repo: Path, target: str = "gfx1030", size: int = 257
) -> tuple[Path, Path, dict[str, object]]:
    run_root = Path(tempfile.mkdtemp(prefix="ullm-h3-artifact-"))
    output = run_root / f"h3-{target}"
    output.mkdir(parents=True)
    artifact_path = output / f"device-code-object-{target}.elf"
    artifact_path.write_bytes(b"x" * size)
    matrix = json.loads((repo / "ci/matrix/hip-compile-v1.json").read_text(encoding="utf-8"))
    toolchain = json.loads((repo / "ci/toolchains/rocm-7.14.0.json").read_text(encoding="utf-8"))
    row = next(row for row in matrix["rows"] if row["target"] == target)
    e_flags = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}[target]
    metadata: dict[str, object] = {
        "schema_version": "hip-artifact-metadata-v1",
        "metadata_id": f"h3-artifact-{target}",
        "matrix_row_id": f"h3-{target}",
        "target": target,
        "candidate": {
            "commit_sha": "a" * 40,
            "tree_oid": "b" * 40,
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
        },
        "toolchain_id": "rocm-7.14.0",
        "matrix_id": "hip-compile-v1",
        "toolchain_manifest_sha256": sha256_json(toolchain),
        "matrix_manifest_sha256": sha256_json(matrix),
        "image": {key: toolchain["image"][key] for key in (
            "repository", "tag", "manifest_digest", "config_digest",
            "manifest_list_digest", "manifest_type", "platform",
        )},
        "resolved_paths": copy.deepcopy(toolchain["paths"]),
        "build": {
            "source_directory": "/workspace/uLLM-project",
            "output_directory": str(output),
            "generator": "Unix Makefiles",
            "build_type": "Release",
            "output_directory_scope": "row-private",
            "source_tree_output": False,
            "shared_build_directory": False,
        },
        "codegen": copy.deepcopy(row["codegen"]),
        "artifact": {
            "path": str(artifact_path),
            "size_bytes": size,
            "sha256": hashlib.sha256(b"x" * size).hexdigest(),
        },
        "host_bundle": {
            "format": "ELF64",
            "machine": "X86_64",
            "bundles": [{
                "target": target,
                "id": f"hipv4-amdgcn-amd-amdhsa--{target}",
            }],
            "sections": {
                ".hip_fatbin": {"present": True, "size_bytes": 256},
            },
        },
        "device_code_object": {
            "format": "ELF64",
            "machine": "AMDGPU",
            "target": target,
            "ei_abiversion": 4,
            "e_flags": e_flags,
            "code_object_version": "V6",
            "wavefront_size": 32,
            "features": {
                "xnack": "unsupported",
                "sramecc": "unsupported",
                "generic_processor_version": 0,
            },
            "sections": {
                ".text": {"present": True, "size_bytes": 255},
            },
            "symbols": [{"name": "ullm_hip_compile_probe", "defined": True}],
        },
        "scope": {
            "compile_only": True,
            "link_verified": True,
            "gpu_execution": False,
            "execution_attempted": False,
            "numerics_verified": False,
            "model_verified": False,
            "performance_verified": False,
            "support_claim": False,
            "network_used": False,
            "model_used": False,
            "cpu_fallback_used": False,
        },
        "timestamps": {
            "created_at": "2026-08-03T05:00:00Z",
            "started_at": "2026-08-03T05:00:00Z",
            "finished_at": "2026-08-03T05:00:01Z",
        },
        "duration_seconds": 1,
    }
    metadata_path = run_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return run_root, metadata_path, metadata


class H3ManifestNegativeTests(unittest.TestCase):
    def assert_manifest_rejected(self, relative: str, mutation, label: str) -> None:
        root = copy_contract_tree()
        try:
            mutate_json(root, relative, mutation)
            with self.subTest(label=label):
                with self.assertRaises(ContractError):
                    validate_h3_contracts(root)
        finally:
            shutil.rmtree(root)

    def test_valid_manifests_and_exact_two_rows(self) -> None:
        root = copy_contract_tree()
        try:
            toolchain, matrix = validate_h3_manifests(root)
            self.assertEqual(toolchain["image"]["manifest_list_digest"], None)
            self.assertEqual([row["target"] for row in matrix["rows"]], list(EXPECTED_TARGETS))
        finally:
            shutil.rmtree(root)

    def test_image_pin_and_platform_fail_closed(self) -> None:
        cases = [
            (lambda doc: doc["image"].pop("manifest_digest"), "tag-only"),
            (lambda doc: doc["image"].update(tag="latest"), "latest"),
            (lambda doc: doc["image"].update(manifest_digest="sha256:" + "0" * 64), "manifest digest mismatch"),
            (lambda doc: doc["image"].update(config_digest="sha256:" + "1" * 64), "config digest mismatch"),
            (lambda doc: doc["image"]["platform"].update(architecture="arm64"), "arm64"),
        ]
        for mutation, label in cases:
            self.assert_manifest_rejected("ci/toolchains/rocm-7.14.0.json", mutation, label)

    def test_rocm_root_and_llvm_fail_closed(self) -> None:
        cases = [
            (lambda doc: doc["rocm"].update(version="7.13.0"), "wrong ROCm"),
            (lambda doc: doc["rocm"].update(llvm_major=22), "wrong LLVM"),
            (lambda doc: doc["rocm"].update(path="/opt/rocm-7.14"), "wrong root"),
            (lambda doc: doc["compiler"].update(path="/usr/bin/amdclang++"), "compiler root"),
        ]
        for mutation, label in cases:
            self.assert_manifest_rejected("ci/toolchains/rocm-7.14.0.json", mutation, label)

    def test_rows_are_exact_unique_and_non_required(self) -> None:
        cases = [
            (lambda doc: doc["rows"].pop(), "missing row"),
            (lambda doc: doc["rows"][1].update(row_id=doc["rows"][0]["row_id"]), "duplicate row"),
            (lambda doc: doc["rows"][1].update(row_id="h3-unknown"), "unknown row"),
            (lambda doc: doc["rows"][1].update(target="gfx12-generic"), "generic target"),
            (lambda doc: doc["rows"][1].update(target=["gfx1201", "gfx1030"]), "multiple target"),
            (lambda doc: doc["rows"][1].update(target="gfx1030"), "wrong target"),
            (lambda doc: doc["rows"][0].update(required=True), "required true"),
            (lambda doc: doc["rows"][0]["codegen"].update(code_object_version="V5"), "CO mismatch"),
            (lambda doc: doc["rows"][0]["codegen"]["features"].update(xnack="off"), "feature mismatch"),
        ]
        for mutation, label in cases:
            self.assert_manifest_rejected("ci/matrix/hip-compile-v1.json", mutation, label)

    def test_schema_is_closed(self) -> None:
        self.assert_manifest_rejected(
            "ci/toolchains/rocm-7.14.0.json",
            lambda doc: doc.update(unexpected=True),
            "toolchain additional property",
        )
        self.assert_manifest_rejected(
            "ci/matrix/hip-compile-v1.json",
            lambda doc: doc["rows"][0]["execution"].update(unexpected=True),
            "matrix nested additional property",
        )


class H3ArtifactNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = copy_contract_tree()
        self.run_root, self.metadata_path, self.metadata = artifact_fixture(self.repo)
        self.base_metadata = copy.deepcopy(
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.repo)
        shutil.rmtree(self.run_root)

    def assert_artifact_rejected(self, mutation, label: str) -> None:
        document = copy.deepcopy(self.base_metadata)
        mutation(document)
        self.metadata_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        with self.subTest(label=label):
            with self.assertRaises(ContractError):
                validate_artifact_metadata(self.metadata_path, self.repo)

    def test_artifact_boundary_sizes_255_256_257_are_valid_when_hashed(self) -> None:
        for size in (255, 256, 257):
            with self.subTest(size=size):
                artifact_path = Path(self.metadata["artifact"]["path"])
                artifact_path.write_bytes(b"z" * size)
                document = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                document["artifact"]["size_bytes"] = size
                document["artifact"]["sha256"] = hashlib.sha256(b"z" * size).hexdigest()
                self.metadata_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                validate_artifact_metadata(self.metadata_path, self.repo)

    def test_gfx1201_target_specific_two_layer_evidence_is_valid(self) -> None:
        run_root, metadata_path, _metadata = artifact_fixture(self.repo, target="gfx1201")
        try:
            validate_artifact_metadata(metadata_path, self.repo)
        finally:
            shutil.rmtree(run_root)

    def test_candidate_sha_tree_and_manifest_hashes_are_bound(self) -> None:
        self.assert_artifact_rejected(
            lambda doc: doc["candidate"].update(reviewed_sha="c" * 40), "candidate SHA disagreement"
        )
        self.assert_artifact_rejected(
            lambda doc: doc["candidate"].update(tree_oid="not-a-tree"), "invalid tree OID"
        )
        self.assert_artifact_rejected(
            lambda doc: doc.update(toolchain_manifest_sha256="0" * 64), "toolchain hash mismatch"
        )
        self.assert_artifact_rejected(
            lambda doc: doc.update(matrix_manifest_sha256="1" * 64), "matrix hash mismatch"
        )
        self.metadata_path.write_text(json.dumps(self.base_metadata, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            validate_artifact_metadata(self.metadata_path, self.repo, expected_candidate_sha="c" * 40)
        with self.assertRaises(ContractError):
            validate_artifact_metadata(self.metadata_path, self.repo, expected_tree_oid="c" * 40)

    def test_artifact_target_codegen_path_and_scope_are_bound(self) -> None:
        cases = [
            (lambda doc: doc.update(target="gfx1201"), "artifact target swap"),
            (lambda doc: doc["codegen"].update(wavefront_size=64), "wave mismatch"),
            (lambda doc: doc["device_code_object"].update(target="gfx1201"), "device target swap"),
            (lambda doc: doc["device_code_object"].update(e_flags="0x0000004e"), "device e_flags swap"),
            (lambda doc: doc["device_code_object"].update(ei_abiversion=5), "device ABI swap"),
            (lambda doc: doc["host_bundle"]["bundles"][0].update(target="gfx1201"), "bundle target swap"),
            (lambda doc: doc["host_bundle"]["bundles"][0].update(
                id="hipv4-amdgcn-amd-amdhsa--gfx1201"
            ), "bundle ID swap"),
            (lambda doc: doc["host_bundle"]["bundles"].append(
                copy.deepcopy(doc["host_bundle"]["bundles"][0])
            ), "multiple bundle"),
            (lambda doc: doc["host_bundle"]["sections"].pop(".hip_fatbin"), "missing .hip_fatbin"),
            (lambda doc: doc["host_bundle"].pop("format"), "missing host ELF format"),
            (lambda doc: doc["host_bundle"].update(format="ELF32"), "invalid host ELF format"),
            (lambda doc: doc["host_bundle"].update(machine="AMDGPU"), "host AMDGPU machine"),
            (lambda doc: doc["host_bundle"].update(machine="HOST"), "abstract host machine"),
            (lambda doc: doc["device_code_object"]["sections"].pop(".text"), "missing device .text"),
            (lambda doc: doc["artifact"].update(path=doc["artifact"]["path"].replace("gfx1030", "gfx1201")), "artifact path swap"),
            (lambda doc: doc["artifact"].update(path=doc["artifact"]["path"].replace(".elf", ".a")), "archive artifact forbidden"),
            (lambda doc: doc["artifact"].update(sha256="f" * 64), "artifact SHA mismatch"),
            (lambda doc: doc["artifact"].update(size_bytes=256), "artifact size mismatch"),
            (lambda doc: doc["build"].update(source_directory=doc["build"]["output_directory"]), "source tree output"),
            (lambda doc: doc["build"].update(output_directory_scope="shared"), "shared output scope"),
            (lambda doc: doc["scope"].update(gpu_execution=True), "GPU execution scope"),
            (lambda doc: doc["scope"].update(model_verified=True), "model scope"),
        ]
        for mutation, label in cases:
            repo = copy_contract_tree()
            run_root, metadata_path, _ = artifact_fixture(repo)
            try:
                document = json.loads(metadata_path.read_text(encoding="utf-8"))
                mutation(document)
                metadata_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                with self.subTest(label=label):
                    with self.assertRaises(ContractError):
                        validate_artifact_metadata(metadata_path, repo)
            finally:
                shutil.rmtree(repo)
                shutil.rmtree(run_root)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("ULLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "ULLM_UNITTEST_COUNTS="
            + json.dumps(
                {
                    "collected": selected,
                    "selected": selected,
                    "passed": selected - failed - skipped,
                    "failed": failed,
                    "skipped": skipped,
                    "deselected": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
