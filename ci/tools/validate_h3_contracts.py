#!/usr/bin/env python3
"""Fail-closed validation for the Phase 2 H3 static contracts.

The default command validates only checked-in toolchain and compile-matrix
manifests.  Artifact metadata is validated when supplied explicitly; doing so
also verifies the referenced artifact's size and SHA-256 without compiling or
executing HIP code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, parse_time, read_json, sha256_file, sha256_json  # noqa: E402

EXPECTED_TARGETS = ("gfx1030", "gfx1201")
EXPECTED_TOOLCHAIN_ID = "rocm-7.14.0"
EXPECTED_MATRIX_ID = "hip-compile-v1"
BUNDLE_IDS = {
    "gfx1030": "hipv4-amdgcn-amd-amdhsa--gfx1030",
    "gfx1201": "hipv4-amdgcn-amd-amdhsa--gfx1201",
}
HOST_BUNDLE_ID = "host-x86_64-unknown-linux-gnu-"
PINNED_IMAGE_REFERENCE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
DEVICE_E_FLAGS = {
    "gfx1030": "0x00000036",
    "gfx1201": "0x0000004e",
}
H3_SEEDS = {"gfx1030": 1030, "gfx1201": 1201}
DIRECT_BUILD = {
    "driver": "/opt/rocm/bin/amdclang++",
    "mode": "direct-compile-link",
    "build_type": "Release",
    "timeout_seconds": 900,
    "source_relative_path": "native/hip/src/hip_compile_probe.hip.cpp",
    "object_pattern": "hip-compile-probe-{target}.o",
    "link_output_pattern": "hip-compile-probe-{target}.elf",
    "commands": [
        ["/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17", "--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64", "-o", "{build_dir}/hip-compile-probe-{target}.o", "-x", "hip", "-c", "{source_path}"],
        ["/opt/rocm/bin/amdclang++", "-O3", "-DNDEBUG", "--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64", "--hip-link", "--rtlib=compiler-rt", "-unwindlib=libgcc", "{build_dir}/hip-compile-probe-{target}.o", "-o", "{build_dir}/hip-compile-probe-{target}.elf", "/opt/rocm/lib/libamdhip64.so"],
    ],
}
IMAGE_KEYS = (
    "repository", "tag", "manifest_digest", "config_digest",
    "manifest_list_digest", "manifest_type", "platform",
)
PATH_KEYS = (
    "rocm_root", "compiler", "hip_headers", "hip_cmake_package",
    "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy",
    "llvm_readobj", "llvm_objdump",
)


def _schema_validator(schema: dict[str, Any], label: str) -> Any:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - host dependency contract
        raise ContractError("jsonschema is required for H3 contract validation") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema uses several exception subclasses
        raise ContractError(f"{label} schema is invalid: {exc}") from exc
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate_schema(document: Any, schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        _schema_validator(schema, label).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{label} {'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ContractError(details)


def _matrix_schema(toolchain_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve matrix $defs while preserving refs to sibling definitions."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ullm-project.local/ci/schema/hip-compile-v1.schema.json",
        "$defs": toolchain_schema["$defs"],
        "$ref": "#/$defs/hip_compile_matrix",
    }


def _load_contract_files(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    toolchain_schema = read_json(repo / "ci/schema/rocm-toolchain-v1.schema.json")
    artifact_schema = read_json(repo / "ci/schema/hip-artifact-metadata-v1.schema.json")
    toolchain = read_json(repo / "ci/toolchains/rocm-7.14.0.json")
    matrix = read_json(repo / "ci/matrix/hip-compile-v1.json")
    if not isinstance(toolchain_schema, dict) or not isinstance(artifact_schema, dict):
        raise ContractError("H3 schemas must be JSON objects")
    if not isinstance(toolchain, dict) or not isinstance(matrix, dict):
        raise ContractError("H3 manifests must be JSON objects")
    return toolchain_schema, artifact_schema, toolchain, matrix


def _path_is_within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    return path == parent or parent in path.parents


def _require_same_mapping(left: dict[str, Any], right: dict[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if left.get(key) != right.get(key):
            raise ContractError(f"{label}.{key} does not match checked-in contract")


def _validate_toolchain_invariants(toolchain: dict[str, Any]) -> None:
    if toolchain["toolchain_id"] != EXPECTED_TOOLCHAIN_ID:
        raise ContractError("toolchain id is not the pinned ROCm 7.14.0 identity")
    image = toolchain["image"]
    if image["tag"] == "latest" or not image["manifest_digest"] or not image["config_digest"]:
        raise ContractError("H3 image requires a non-latest tag and both immutable digests")
    if image["manifest_list_digest"] is not None:
        raise ContractError("single-manifest image must use manifest_list_digest=null")
    if image["platform"] != {"os": "linux", "architecture": "amd64"}:
        raise ContractError("H3 image platform must be exactly linux/amd64")
    root = PurePosixPath(toolchain["rocm"]["path"])
    if str(root) != "/opt/rocm":
        raise ContractError("ROCm root is not canonical /opt/rocm")
    if toolchain["rocm"]["version"] != "7.14.0" or toolchain["rocm"]["llvm_major"] != 23:
        raise ContractError("ROCm release or LLVM major does not match the pinned toolchain")
    if toolchain["compiler"]["llvm_major"] != toolchain["rocm"]["llvm_major"]:
        raise ContractError("compiler LLVM major does not match the ROCm LLVM major")
    if toolchain["compiler"]["path"] != toolchain["paths"]["compiler"]:
        raise ContractError("compiler path is not the resolved compiler path")
    for key in PATH_KEYS:
        value = PurePosixPath(toolchain["paths"][key])
        if value != root and root not in value.parents:
            raise ContractError(f"resolved path is outside the canonical ROCm root: {key}")
    for key in ("clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump"):
        if PurePosixPath(toolchain["paths"][key]).parent.name != "bin":
            raise ContractError(f"LLVM inspector is not in the ROCm LLVM bin directory: {key}")


def _validate_matrix_invariants(matrix: dict[str, Any], toolchain: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if matrix["matrix_id"] != EXPECTED_MATRIX_ID or matrix["toolchain_id"] != toolchain["toolchain_id"]:
        raise ContractError("HIP matrix is not bound to the pinned toolchain")
    if matrix["targets"] != list(EXPECTED_TARGETS):
        raise ContractError("HIP matrix target set is not exactly gfx1030/gfx1201")
    rows = matrix["rows"]
    if len(rows) != 2:
        raise ContractError("H3 HIP matrix must contain exactly two rows")
    by_row: dict[str, dict[str, Any]] = {}
    by_target: dict[str, str] = {}
    for row in rows:
        row_id = row["row_id"]
        target = row["target"]
        if row_id in by_row:
            raise ContractError(f"duplicate H3 row: {row_id}")
        if target in by_target:
            raise ContractError(f"duplicate H3 target: {target}")
        if target not in EXPECTED_TARGETS or row_id != f"h3-{target}":
            raise ContractError(f"H3 row identity does not match its exact target: {row_id}/{target}")
        if row["seed"] != H3_SEEDS[target]:
            raise ContractError(f"H3 row seed is not bound to its exact target: {row_id}")
        if row["tier"] != "tier_h3" or row["required"] is not False:
            raise ContractError(f"H3 row is not explicitly non-required: {row_id}")
        execution = row["execution"]
        if execution != {
            "mode": "compile-only",
            "requires_gpu": False,
            "requires_model": False,
            "network": False,
            "fallback_allowed": False,
        }:
            raise ContractError(f"H3 row has an execution capability outside compile-only scope: {row_id}")
        if row["direct_build"] != DIRECT_BUILD:
            raise ContractError(f"H3 row has a non-canonical direct amdclang++ contract: {row_id}")
        if row["resource"] != {"max_rss_bytes": 4294967296, "max_output_bytes": 16777216}:
            raise ContractError(f"H3 row has non-canonical resource limits: {row_id}")
        if row["output"] != {
            "root_prefix": "/tmp/ullm-h3-",
            "directory_pattern": "h3-{target}",
            "artifact_pattern": "device-code-object-{target}.elf",
        }:
            raise ContractError(f"H3 row has non-canonical private output contract: {row_id}")
        codegen = row["codegen"]
        if codegen["target"] != target or codegen["target_kind"] != "exact" or codegen["target_count"] != 1:
            raise ContractError(f"H3 row target is missing, generic, or multi-target: {row_id}")
        if codegen["code_object_version"] != "V6" or codegen["wavefront_size"] != 32:
            raise ContractError(f"H3 row code object or wavefront is not V6/wave32: {row_id}")
        if codegen["features"] != {
            "xnack": "unsupported",
            "sramecc": "unsupported",
            "generic_processor_version": 0,
        }:
            raise ContractError(f"H3 row codegen features are not the measured unsupported tuple: {row_id}")
        by_row[row_id] = row
        by_target[target] = row_id
    if set(by_row) != {f"h3-{target}" for target in EXPECTED_TARGETS}:
        raise ContractError("H3 matrix is missing or has an unknown exact target row")
    return by_row


def validate_h3_manifests(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the checked-in toolchain and HIP matrix and return both documents."""

    toolchain_schema, _artifact_schema, toolchain, matrix = _load_contract_files(repo)
    _validate_schema(toolchain, toolchain_schema, "ROCm toolchain")
    _validate_schema(matrix, _matrix_schema(toolchain_schema), "HIP compile matrix")
    _validate_toolchain_invariants(toolchain)
    _validate_matrix_invariants(matrix, toolchain)
    return toolchain, matrix


def _validate_artifact_invariants(
    metadata: dict[str, Any],
    toolchain: dict[str, Any],
    matrix: dict[str, Any],
    *,
    expected_candidate_sha: str | None = None,
    expected_tree_oid: str | None = None,
    artifact_path_override: Path | None = None,
) -> None:
    rows = {row["row_id"]: row for row in matrix["rows"]}
    row_id = metadata["matrix_row_id"]
    row = rows.get(row_id)
    if row is None:
        raise ContractError(f"artifact metadata references an unknown H3 row: {row_id}")
    target = row["target"]
    if metadata["target"] != target or metadata["metadata_id"] != f"h3-artifact-{target}":
        raise ContractError("artifact metadata target/row identity mismatch")
    if metadata["toolchain_id"] != toolchain["toolchain_id"] or metadata["matrix_id"] != matrix["matrix_id"]:
        raise ContractError("artifact metadata is bound to the wrong toolchain or matrix")
    if metadata["toolchain_manifest_sha256"] != sha256_json(toolchain):
        raise ContractError("artifact toolchain manifest hash is stale or mismatched")
    if metadata["matrix_manifest_sha256"] != sha256_json(matrix):
        raise ContractError("artifact matrix manifest hash is stale or mismatched")
    candidate = metadata["candidate"]
    if len({candidate["commit_sha"], candidate["reviewed_sha"], candidate["tested_sha"], candidate["workflow_sha"]}) != 1:
        raise ContractError("artifact candidate/reviewed/tested/workflow SHA identities disagree")
    if expected_candidate_sha is not None and candidate["commit_sha"] != expected_candidate_sha:
        raise ContractError("artifact candidate SHA is stale")
    if expected_tree_oid is not None and candidate["tree_oid"] != expected_tree_oid:
        raise ContractError("artifact candidate tree OID is stale")
    _require_same_mapping(metadata["image"], toolchain["image"], IMAGE_KEYS, "artifact image")
    _require_same_mapping(metadata["resolved_paths"], toolchain["paths"], PATH_KEYS, "artifact resolved path")
    if metadata["codegen"] != row["codegen"]:
        raise ContractError("artifact codegen does not match its matrix row")
    host_bundle = metadata["host_bundle"]
    if host_bundle["format"] != "ELF64" or host_bundle["machine"] != "X86_64":
        raise ContractError("host bundle ELF identity must be ELF64/X86_64")
    expected_bundles = [
        {"id": BUNDLE_IDS[target], "target": target},
        {"id": HOST_BUNDLE_ID, "target": "host"},
    ]
    if host_bundle["bundles"] != expected_bundles:
        raise ContractError("host bundle list is not the exact device/host order")
    if not host_bundle["sections"][".hip_fatbin"]["present"]:
        raise ContractError("host bundle evidence does not prove .hip_fatbin")

    environment = metadata["execution_environment"]
    if environment["mode"] == "required-ci":
        expected_environment = {
            "mode": "required-ci",
            "execution_scope": "official-container",
            "container_image_reference": PINNED_IMAGE_REFERENCE,
            "observed_image_config_digest": "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf",
            "pinned_container": True,
            "identity_verified": True,
            "network_isolated": True,
        }
    else:
        expected_environment = {
            "mode": "local-development",
            "execution_scope": "local-system",
            "container_image_reference": None,
            "observed_image_config_digest": None,
            "pinned_container": False,
            "identity_verified": False,
            "network_isolated": False,
        }
    if environment != expected_environment:
        raise ContractError("H3 execution environment does not match its evidence mode")

    device = metadata["device_code_object"]
    if (
        device["format"] != "ELF64"
        or device["machine"] != "AMDGPU"
        or device["target"] != target
        or device["ei_abiversion"] != 4
    ):
        raise ContractError("device code object ELF identity or ABI does not match the exact H3 target")
    if device["e_flags"] != DEVICE_E_FLAGS[target]:
        raise ContractError("device code object e_flags does not match the measured target value")
    if device["code_object_version"] != "V6" or device["wavefront_size"] != 32:
        raise ContractError("device code object is not V6/wave32")
    if device["features"] != {
        "xnack": "unsupported",
        "sramecc": "unsupported",
        "generic_processor_version": 0,
    }:
        raise ContractError("device code object features do not match the measured unsupported tuple")
    if not device["sections"][".text"]["present"]:
        raise ContractError("device code object does not prove .text")
    if not device["symbols"] or not any(
        symbol["name"] == "ullm_hip_compile_probe" and symbol["defined"]
        for symbol in device["symbols"]
    ):
        raise ContractError("device code object has no defined compile-probe symbol")
    build = metadata["build"]
    source = PurePosixPath(build["source_directory"])
    source_path = PurePosixPath(build["source_path"])
    output = PurePosixPath(build["output_directory"])
    if _path_is_within(output, source):
        raise ContractError("H3 output directory is inside the source tree")
    if source_path != source / row["direct_build"]["source_relative_path"]:
        raise ContractError("H3 direct compile source path does not match its row contract")
    if output.name != f"h3-{target}" or build["output_directory_scope"] != "row-private":
        raise ContractError("H3 output directory is not private to its exact row")
    expected_object = output / row["direct_build"]["object_pattern"].replace("{target}", target)
    expected_link = output / row["direct_build"]["link_output_pattern"].replace("{target}", target)
    if PurePosixPath(build["object_path"]) != expected_object or PurePosixPath(build["link_output_path"]) != expected_link:
        raise ContractError("H3 direct compile/link outputs do not match the exact target row")
    if build["generator"] != "direct-amdclang++" or build["mode"] != "direct-compile-link" or build["build_type"] != "Release" or build["language_standard"] != "gnu++17":
        raise ContractError("H3 metadata build record does not identify direct amdclang++ Release compilation")
    if build["source_tree_output"] is not False or build["shared_build_directory"] is not False:
        raise ContractError("H3 source-tree/shared-build output is forbidden")
    metadata_artifact_path = PurePosixPath(metadata["artifact"]["path"])
    if metadata_artifact_path.parent != output or not metadata_artifact_path.name.endswith(f"-{target}.elf"):
        raise ContractError("artifact path is outside the row output directory or is not the exact device ELF target")
    if any(f"-gfx{other[3:]}" in metadata_artifact_path.name for other in EXPECTED_TARGETS if other != target):
        raise ContractError("artifact path contains another H3 target identity")
    actual_path = artifact_path_override or Path(str(metadata_artifact_path))
    if not actual_path.is_absolute():
        raise ContractError("bound H3 artifact path is not absolute")
    if artifact_path_override is not None and (
        actual_path.name != metadata_artifact_path.name
        or actual_path.parent.name != f"h3-{target}"
    ):
        raise ContractError("rebound H3 artifact is not the exact staged target artifact")
    if not actual_path.exists() or not actual_path.is_file() or actual_path.is_symlink():
        raise ContractError(f"artifact file is missing or not a regular non-symlink file: {actual_path}")
    actual_size = actual_path.stat().st_size
    if actual_size != metadata["artifact"]["size_bytes"]:
        raise ContractError("artifact size metadata is stale")
    if sha256_file(actual_path) != metadata["artifact"]["sha256"]:
        raise ContractError("artifact content SHA-256 does not match metadata")
    timestamps = metadata["timestamps"]
    created = parse_time(timestamps["created_at"])
    started = parse_time(timestamps["started_at"])
    finished = parse_time(timestamps["finished_at"])
    if not created <= started <= finished:
        raise ContractError("artifact timestamps are not ordered")
    elapsed = (finished - started).total_seconds()
    if metadata["duration_seconds"] > elapsed + 0.001:
        raise ContractError("artifact duration exceeds its timestamps")


def validate_artifact_metadata(
    metadata_path: Path,
    repo: Path = ROOT,
    *,
    expected_candidate_sha: str | None = None,
    expected_tree_oid: str | None = None,
    artifact_path_override: Path | None = None,
) -> dict[str, Any]:
    """Validate metadata against its declared artifact or an exact staged copy.

    A detached artifact may be supplied only when the immutable metadata remains
    unchanged and the staged filename/row identity still match the declaration.
    """

    toolchain_schema, artifact_schema, toolchain, matrix = _load_contract_files(repo)
    _validate_schema(toolchain, toolchain_schema, "ROCm toolchain")
    _validate_schema(matrix, _matrix_schema(toolchain_schema), "HIP compile matrix")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ContractError("HIP artifact metadata must be a JSON object")
    _validate_schema(metadata, artifact_schema, "HIP artifact metadata")
    _validate_toolchain_invariants(toolchain)
    _validate_matrix_invariants(matrix, toolchain)
    _validate_artifact_invariants(
        metadata,
        toolchain,
        matrix,
        expected_candidate_sha=expected_candidate_sha,
        expected_tree_oid=expected_tree_oid,
        artifact_path_override=artifact_path_override,
    )
    return metadata


def validate_h3_contracts(
    repo: Path = ROOT,
    artifact_metadata_paths: Iterable[Path] = (),
    *,
    expected_candidate_sha: str | None = None,
    expected_tree_oid: str | None = None,
) -> None:
    """Validate all static H3 contracts, optionally including artifact metadata."""

    validate_h3_manifests(repo)
    for metadata_path in artifact_metadata_paths:
        validate_artifact_metadata(
            Path(metadata_path),
            repo,
            expected_candidate_sha=expected_candidate_sha,
            expected_tree_oid=expected_tree_oid,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifact-metadata", type=Path, action="append", default=[])
    parser.add_argument("--expected-candidate-sha", type=str)
    parser.add_argument("--expected-tree-oid", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_h3_contracts(
            args.repo,
            args.artifact_metadata,
            expected_candidate_sha=args.expected_candidate_sha,
            expected_tree_oid=args.expected_tree_oid,
        )
    except (ContractError, OSError, ValueError) as exc:
        print(f"H3 contract validation: FAIL: {exc}", file=sys.stderr)
        return 1
    print("H3 contract validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
