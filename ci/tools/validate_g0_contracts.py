#!/usr/bin/env python3
"""Validate Phase 2 trusted-local G0 preflight contracts without using a GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    exact_sha,
    parse_time,
    read_json,
    sha256_file,
    sha256_json,
)
from validate_h3_contracts import validate_artifact_metadata  # noqa: E402

EXPECTED_ROWS = (
    {
        "row_id": "g0-gfx1030",
        "target": "gfx1030",
        "bdf": "0000:03:00.0",
        "uuid": "GPU-76a08c022586fed6",
        "product": "AMD Radeon Pro V620",
        "h3_artifact_row_id": "h3-gfx1030",
        "rocm": {
            "root": "/opt/rocm/core-7.14",
            "release": "7.14.0",
            "hip_runtime_api_version": 71460850,
            "hip_runtime_library": "/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000",
            "hsa_runtime_library": "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0",
        },
        "timeout_seconds": 300,
        "seed": 1031,
    },
    {
        "row_id": "g0-gfx1201",
        "target": "gfx1201",
        "bdf": "0000:47:00.0",
        "uuid": "GPU-a8e9ddefa2d60f55",
        "product": "AMD Radeon AI PRO R9700",
        "h3_artifact_row_id": "h3-gfx1201",
        "rocm": {
            "root": "/opt/rocm/core-7.14",
            "release": "7.14.0",
            "hip_runtime_api_version": 71460850,
            "hip_runtime_library": "/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000",
            "hsa_runtime_library": "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0",
        },
        "timeout_seconds": 300,
        "seed": 1202,
    },
)
EXPECTED_EXECUTION = {
    "serial": True,
    "host_lock": {"path": "/tmp/ullm-g0.lock", "acquisition": "nonblocking"},
    "trusted_local_only": True,
    "visibility_is_security_boundary": False,
    "sudo_allowed": False,
    "reset_allowed": False,
    "credentials_allowed": False,
    "docker_socket_allowed": False,
    "native_observation_provider": {
        "provider_id": "g0-native-hip-observer-v1",
        "source": "ci/tools/g0_native_observer.cpp",
        "compiler": "/opt/rocm/core-7.14/bin/amdclang++",
        "output_prefix": "/tmp/ullm-g0-provider-",
        "timeout_seconds": 60,
        "allowed_hip_apis": [
            "hipRuntimeGetVersion",
            "hipGetDeviceCount",
            "hipGetDeviceProperties",
            "hipDeviceGetPCIBusId",
            "hipDeviceGetUuid",
        ],
    },
    "health_process_observer": {
        "provider_id": "amd-smi-sysfs-read-only-v1",
        "amd_smi": "/opt/rocm/core-7.14/bin/amd-smi",
        "sysfs_pci_root": "/sys/bus/pci/devices",
        "command_timeout_seconds": 30,
        "no_process_sentinel": "No running processes detected",
    },
}
EXPECTED_SCOPE = {
    "identity_probe_only": True,
    "model_used": False,
    "allocation_attempted": False,
    "copy_attempted": False,
    "kernel_attempted": False,
    "dispatch_attempted": False,
    "native_hip_observation_provider": "native-hip-observer-v1",
    "execution_verified": False,
    "numerics_verified": False,
    "performance_verified": False,
    "support_claim": False,
}
EXPECTED_OUTPUT = {
    "root_prefix": "/tmp/ullm-g0-",
    "directory_pattern": "g0-{target}",
    "source_tree_output": False,
    "sidecar_hashes": True,
}
VISIBILITY_NAMES = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
GPU_SELECTOR_NAMES = (*VISIBILITY_NAMES, "ROCR_VISIBLE_DEVICES")
UUID_TOKEN = re.compile(r"^GPU-[0-9a-f]{16}$")
BDF_TOKEN = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
INDEX_TOKEN = re.compile(r"^(?:0|[1-9][0-9]*)$")
GCN_ARCH_TOKEN = re.compile(r"^(gfx[0-9a-f]+)(?::[A-Za-z0-9_+\-]+)*$")
HIP_UUID_HEX_TOKEN = re.compile(r"^[0-9a-f]{16}$")
AMD_SMI_UUID_TOKEN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
NATIVE_PROVIDER_SOURCE = "ci/tools/g0_native_observer.cpp"
AMD_SMI_EXECUTABLE = "/opt/rocm/core-7.14/bin/amd-smi"
AMD_SMI_LIST_COMMAND = (AMD_SMI_EXECUTABLE, "list", "-e", "--json")
ALLOWED_HIP_PROVIDER_APIS = {
    "hipRuntimeGetVersion",
    "hipGetDeviceCount",
    "hipGetDeviceProperties",
    "hipDeviceGetPCIBusId",
    "hipDeviceGetUuid",
}
FORBIDDEN_HIP_PROVIDER_APIS = (
    "hipMalloc",
    "hipMallocManaged",
    "hipMallocAsync",
    "hipHostAlloc",
    "hipHostMalloc",
    "hipMallocHost",
    "hipFree",
    "hipFreeHost",
    "hipMemcpy",
    "hipMemset",
    "hipLaunchKernel",
    "hipModuleLaunchKernel",
    "hipExtLaunchKernel",
    "hipGraphLaunch",
    "hipDeviceSynchronize",
    "hipStreamCreate",
    "hipEventCreate",
)
HIP_PROVIDER_CALL = re.compile(r"\b(hip[A-Za-z0-9_]*)\s*\(")


def schema_validator(schema: dict[str, Any], label: str) -> Any:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - pinned host dependency
        raise ContractError("jsonschema is required for G0 contract validation") from exc
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_schema(document: Any, schema: dict[str, Any], label: str) -> None:
    errors = sorted(schema_validator(schema, label).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ContractError(f"{label} schema validation failed: {detail}")


def matrix_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/gpu_runtime_matrix",
    }


def preflight_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/preflight",
    }


def load_g0_contract(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = read_json(repo / "ci/schema/g0-preflight-v1.schema.json")
    matrix = read_json(repo / "ci/matrix/gpu-runtime-v1.json")
    if not isinstance(schema, dict) or not isinstance(matrix, dict):
        raise ContractError("G0 schema and matrix must be JSON objects")
    validate_schema(matrix, matrix_schema(schema), "G0 runtime matrix")
    return schema, matrix


def validate_g0_matrix(repo: Path = ROOT) -> dict[str, Any]:
    """Require the exact two serial canonical rows and non-execution scope."""

    _schema, matrix = load_g0_contract(repo)
    if matrix.get("matrix_id") != "gpu-runtime-v1" or matrix.get("revision") != 2:
        raise ContractError("G0 matrix identity/revision drifted")
    if matrix.get("tier") != "tier_g0" or matrix.get("required") is not True:
        raise ContractError("G0 matrix must be a required tier_g0 contract")
    if matrix.get("toolchain_id") != "rocm-7.14.0" or matrix.get("artifact_matrix_id") != "hip-compile-v1":
        raise ContractError("G0 matrix is not linked to the H3 ROCm/artifact contract")
    if matrix.get("execution") != EXPECTED_EXECUTION:
        raise ContractError("G0 serial lock/security contract drifted")
    if matrix.get("scope") != EXPECTED_SCOPE:
        raise ContractError("G0 must remain an identity-only non-execution preflight")
    if matrix.get("output") != EXPECTED_OUTPUT:
        raise ContractError("G0 output/sidecar contract drifted")
    if matrix.get("rows") != list(EXPECTED_ROWS):
        raise ContractError("G0 rows must be exactly the ordered canonical gfx1030/gfx1201 pair")
    native_provider_source_contract(repo)
    return matrix


def validate_native_provider_source_text(source: str) -> None:
    """Reject every HIP API that can allocate, copy, or submit GPU work.

    This deliberately scans comments and string literals too.  The provider is
    tiny and must not even document a forbidden call in its implementation,
    which keeps the source-level contract simple and fail-closed.
    """

    if not isinstance(source, str) or not source:
        raise ContractError("native HIP observation provider source is empty or malformed")
    for api in FORBIDDEN_HIP_PROVIDER_APIS:
        if re.search(rf"\b{re.escape(api)}(?:Async)?\b", source):
            raise ContractError(f"native HIP observation provider uses forbidden API: {api}")
    for api in HIP_PROVIDER_CALL.findall(source):
        if api not in ALLOWED_HIP_PROVIDER_APIS:
            raise ContractError(f"native HIP observation provider uses non-identity HIP API: {api}")
    if "__global__" in source or "<<<" in source:
        raise ContractError("native HIP observation provider contains a kernel definition or launch")


def native_provider_source_contract(repo: Path = ROOT) -> dict[str, str]:
    source_path = repo / NATIVE_PROVIDER_SOURCE
    if not source_path.is_file() or source_path.is_symlink():
        raise ContractError("native HIP observation provider source is missing or unsafe")
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read native HIP observation provider source: {exc}") from exc
    validate_native_provider_source_text(source)
    return {
        "source_path": NATIVE_PROVIDER_SOURCE,
        "source_sha256": sha256_file(source_path),
    }


def exact_target_from_gcn_arch_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError("HIP gcnArchName is missing or not a string")
    match = GCN_ARCH_TOKEN.fullmatch(value)
    if match is None:
        raise ContractError("HIP gcnArchName is malformed or loses exact feature text")
    return match.group(1)


def canonical_uuid_from_hip_bytes(value: Any) -> str:
    """HIP 7.14 exposes the UUID struct as sixteen ASCII hex characters."""

    if not isinstance(value, str) or HIP_UUID_HEX_TOKEN.fullmatch(value) is None:
        raise ContractError("HIP UUID hexadecimal value is missing or malformed")
    return f"GPU-{value.lower()}"


def amd_smi_uuid_to_hip_uuid(value: Any) -> str:
    """Convert AMD-SMI's UUID layout to the HIP UUID shown by list -e."""

    if not isinstance(value, str) or AMD_SMI_UUID_TOKEN.fullmatch(value) is None:
        raise ContractError("AMD-SMI UUID is missing or malformed")
    compact = value.replace("-", "").lower()
    return f"GPU-{compact[:2]}{compact[18:]}"


def row_by_id(matrix: dict[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in matrix["rows"] if row["row_id"] == row_id]
    if len(matches) != 1:
        raise ContractError(f"G0 row is missing, duplicate, or unknown: {row_id}")
    return matches[0]


def validate_candidate(candidate: Mapping[str, Any], *, expected_sha: str | None = None, expected_tree: str | None = None) -> None:
    required = {"reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "worktree_clean", "revision_input"}
    if set(candidate) != required:
        raise ContractError("candidate identity has missing or unknown fields")
    values = [exact_sha(candidate[name], name) for name in ("reviewed_sha", "tested_sha", "workflow_sha")]
    exact_sha(candidate["git_tree_oid"], "git_tree_oid")
    if len(set(values)) != 1:
        raise ContractError("reviewed/tested/workflow SHA values must match exactly")
    if candidate["worktree_clean"] is not True:
        raise ContractError("G0 rejects a dirty candidate")
    if candidate["revision_input"] != "full-sha":
        raise ContractError("G0 rejects branch, tag, ref, and other mutable revision input")
    if expected_sha is not None and values[0] != exact_sha(expected_sha, "expected_sha"):
        raise ContractError("candidate SHA does not match the checked-out immutable commit")
    if expected_tree is not None and candidate["git_tree_oid"] != exact_sha(expected_tree, "expected_tree"):
        raise ContractError("candidate tree does not match the checked-out immutable tree")


def normalize_visibility_token(value: str) -> str:
    if value != value.strip() or not value or "," in value or any(character.isspace() for character in value):
        raise ContractError("visibility selector must contain exactly one normalized token")
    if UUID_TOKEN.fullmatch(value) or BDF_TOKEN.fullmatch(value) or INDEX_TOKEN.fullmatch(value):
        return value
    raise ContractError("visibility selector is malformed or uses an ambiguous alias")


def validate_visibility_environment(environment: Mapping[str, str | None]) -> dict[str, str | None]:
    """Reject multiple aliases; a surviving selector remains only a routing hint."""

    if environment.get("ROCR_VISIBLE_DEVICES") is not None:
        raise ContractError("ROCR_VISIBLE_DEVICES is not an allowed G0 visibility field")
    values = {name: environment.get(name) for name in VISIBILITY_NAMES}
    present = [(name, value) for name, value in values.items() if value is not None]
    if len(present) > 1:
        raise ContractError("multiple visibility variables are set; aliases are conflicting by contract")
    if present:
        name, value = present[0]
        assert value is not None
        if not isinstance(value, str):
            raise ContractError(f"{name} visibility selector must be a string")
        values[name] = normalize_visibility_token(value)
    return {**values, "security_boundary": False}


def reject_inherited_visibility_selectors(environment: Mapping[str, Any]) -> None:
    """Reject every known GPU selector before canonical AMD-SMI routing."""

    inherited = [name for name in GPU_SELECTOR_NAMES if environment.get(name) is not None]
    if inherited:
        raise ContractError(
            "G0 rejects inherited GPU visibility selectors before canonical routing: "
            + ", ".join(inherited)
        )


def validate_routing(
    routing: Mapping[str, Any],
    visibility: Mapping[str, str | None],
    row: Mapping[str, Any],
) -> None:
    """Bind the sole HIP visibility hint to AMD-SMI's canonical BDF lookup."""

    expected_command = list(AMD_SMI_LIST_COMMAND)
    if (
        routing["source"] != "amd-smi-list-e-json-v1"
        or routing["amd_smi"] != AMD_SMI_EXECUTABLE
        or routing["argv"] != expected_command
    ):
        raise ContractError("G0 routing did not use the canonical AMD-SMI list -e JSON command")
    if routing["bdf"] != row["bdf"] or routing["uuid"] != row["uuid"]:
        raise ContractError("G0 routing is not bound to the canonical BDF/UUID")
    for field in ("gpu", "hip_id"):
        value = routing[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"G0 routing {field} is malformed")
    expected_visibility = {
        "HIP_VISIBLE_DEVICES": str(routing["hip_id"]),
        "CUDA_VISIBLE_DEVICES": None,
        "GPU_DEVICE_ORDINAL": None,
        "security_boundary": False,
    }
    if dict(visibility) != expected_visibility:
        raise ContractError("G0 must use only the matching AMD-SMI hip_id as its HIP routing hint")


def regular_non_symlink(path: Path, label: str) -> None:
    if not path.is_absolute() or not path.exists() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be an existing absolute regular non-symlink file")


def validate_sidecar(sidecar: Path, target: Path, label: str) -> str:
    regular_non_symlink(sidecar, label)
    expected = f"{sha256_file(target)}  {target.name}\n"
    try:
        actual = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if actual != expected:
        raise ContractError(f"{label} is missing, stale, malformed, or names another file")
    return sha256_file(sidecar)


def path_outside_repo(path: Path, repo: Path, label: str) -> None:
    try:
        path.resolve().relative_to(repo.resolve())
    except ValueError:
        return
    raise ContractError(f"{label} must be outside the source tree")


def validate_artifact_binding(
    binding: Mapping[str, Any],
    row: Mapping[str, Any],
    repo: Path,
    candidate: Mapping[str, Any],
) -> None:
    metadata_path = Path(binding["metadata_path"])
    artifact_path = Path(binding["artifact_path"])
    for path, label in ((metadata_path, "H3 metadata"), (artifact_path, "H3 artifact")):
        regular_non_symlink(path, label)
        path_outside_repo(path, repo, label)
    metadata_path = metadata_path.resolve()
    artifact_path = artifact_path.resolve()
    metadata_sidecar = metadata_path.with_name(metadata_path.name + ".sha256")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    if (
        metadata_path.name != "hip-artifact-metadata.json"
        or metadata_path.parent.name != row["h3_artifact_row_id"]
    ):
        raise ContractError("G0 metadata must be staged in its exact H3 row directory")
    expected_artifact_path = metadata_path.parent / f"device-code-object-{row['target']}.elf"
    if artifact_path != expected_artifact_path:
        raise ContractError("G0 artifact rebinding must use the exact staged H3 device artifact")
    metadata = validate_artifact_metadata(
        metadata_path,
        repo,
        expected_candidate_sha=candidate["reviewed_sha"],
        expected_tree_oid=candidate["git_tree_oid"],
        artifact_path_override=artifact_path,
    )
    if (
        metadata.get("matrix_row_id") != row["h3_artifact_row_id"]
        or metadata.get("target") != row["target"]
    ):
        raise ContractError("G0 artifact metadata is bound to the wrong exact target row")
    metadata_sidecar_hash = validate_sidecar(metadata_sidecar, metadata_path, "H3 metadata sidecar")
    artifact_sidecar_hash = validate_sidecar(artifact_sidecar, artifact_path, "H3 artifact sidecar")
    expected = {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_sidecar_path": str(metadata_sidecar),
        "metadata_sidecar_sha256": metadata_sidecar_hash,
        "metadata_declared_artifact_path": metadata["artifact"]["path"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_sidecar_path": str(artifact_sidecar),
        "artifact_sidecar_sha256": artifact_sidecar_hash,
        "h3_matrix_row_id": row["h3_artifact_row_id"],
        "target": row["target"],
        "toolchain_id": "rocm-7.14.0",
        "toolchain_manifest_sha256": metadata["toolchain_manifest_sha256"],
    }
    if dict(binding) != expected:
        raise ContractError("G0 artifact binding is stale, substituted, or linked to another H3 row")
    if not Path(metadata["artifact"]["path"]).is_absolute():
        raise ContractError("G0 H3 metadata artifact path is not absolute")


def validate_health(observation: Mapping[str, Any], label: str, row: Mapping[str, Any]) -> None:
    if observation["available"] is not True or observation["reliable"] is not True:
        raise ContractError(f"{label} health data is unavailable or unreliable")
    if {
        "bdf": observation["bdf"],
        "uuid": observation["uuid"],
        "gcnArchName": observation["gcnArchName"],
    } != {"bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"]}:
        raise ContractError(f"{label} health data is not bound to the canonical device")
    if observation["source"] != "amd-smi-sysfs-read-only-v1":
        raise ContractError(f"{label} health data does not use the trusted read-only observer")
    parse_time(observation["observed_at"])
    facts = observation["facts"]
    if facts["device_state"] != "active":
        raise ContractError(f"{label} device state is not active")
    if facts["amdgpu_driver_bound"] is not True or facts["runtime_status"] != "active":
        raise ContractError(f"{label} device is not actively bound to amdgpu")
    if any(
        facts[name] is None
        for name in (
            "ras_uncorrectable_count",
            "sysfs_ras_uncorrectable_count",
            "temperature_c",
        )
    ):
        raise ContractError(f"{label} health facts are incomplete")
    if any(
        isinstance(facts[name], bool)
        or not isinstance(facts[name], int)
        or facts[name] < 0
        for name in ("ras_uncorrectable_count", "sysfs_ras_uncorrectable_count")
    ):
        raise ContractError(f"{label} RAS health counters are malformed")
    try:
        finite_temperature = math.isfinite(float(facts["temperature_c"]))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} temperature is not numeric") from exc
    if not finite_temperature:
        raise ContractError(f"{label} temperature is not finite")


def validate_processes(observation: Mapping[str, Any], label: str, row: Mapping[str, Any]) -> None:
    if observation["available"] is not True or observation["reliable"] is not True:
        raise ContractError(f"{label} process data is unavailable or unreliable")
    if {
        "bdf": observation["bdf"],
        "uuid": observation["uuid"],
        "gcnArchName": observation["gcnArchName"],
    } != {"bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"]}:
        raise ContractError(f"{label} process data is not bound to the canonical device")
    if observation["source"] != "amd-smi-sysfs-read-only-v1":
        raise ContractError(f"{label} process data does not use the trusted read-only observer")
    parse_time(observation["observed_at"])
    if observation["gpu_processes"]:
        raise ContractError(f"{label} reports a GPU process on the locked canonical device")
    if observation["residual_runner_children"]:
        raise ContractError(f"{label} reports residual runner children")


def validate_observations(
    preflight: Mapping[str, Any], row: Mapping[str, Any], repo: Path = ROOT
) -> None:
    provider = preflight["provider"]
    source_contract = native_provider_source_contract(repo)
    expected_provider = {
        "provider_id": "g0-native-hip-observer-v1",
        "available": True,
        "source_path": source_contract["source_path"],
        "source_sha256": source_contract["source_sha256"],
        "compiler_path": "/opt/rocm/core-7.14/bin/amdclang++",
    }
    for key, expected in expected_provider.items():
        if provider.get(key) != expected:
            raise ContractError(f"native observation provider {key} is missing, stale, or mismatched")
    if (
        not isinstance(provider.get("compiler_version"), str)
        or not re.search(r"(?:AMD )?clang version 23\.", provider["compiler_version"])
        or provider.get("binary_path") is not None
        or provider.get("binary_removed") is not True
        or not isinstance(provider.get("binary_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provider["binary_sha256"])
        or not isinstance(provider.get("compile_command_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provider["compile_command_sha256"])
        or not isinstance(provider.get("runtime_command_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provider["runtime_command_sha256"])
    ):
        raise ContractError("native observation provider provenance is malformed")
    device = preflight["device"]
    if device["observed"] is not True:
        raise ContractError("HIP identity probe is unavailable")
    if device["probe_kind"] != "hip-identity-only-v1":
        raise ContractError("HIP observation provider has the wrong probe kind")
    if device["visible_device_count"] != 1 or device["ordinal"] != 0:
        raise ContractError("G0 requires exactly one visible canonical HIP device at ordinal zero")
    if {
        "bdf": device["bdf"],
        "uuid": device["uuid"],
        "product": device["product"],
        "rocm_root": device["rocm_root"],
    } != {
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "product": row["product"],
        "rocm_root": row["rocm"]["root"],
    }:
        raise ContractError("HIP identity probe does not match exact BDF/UUID/product/ROCm root")
    if exact_target_from_gcn_arch_name(device["gcnArchName"]) != row["target"]:
        raise ContractError("HIP gcnArchName does not select the canonical exact target")
    if device["exact_target"] != row["target"]:
        raise ContractError("HIP exact target is not the canonical target")
    if canonical_uuid_from_hip_bytes(device["hip_uuid_hex"]) != row["uuid"]:
        raise ContractError("HIP UUID hexadecimal value does not derive the canonical UUID")
    if (
        isinstance(device["wave_size"], bool)
        or not isinstance(device["wave_size"], int)
        or isinstance(device["total_global_memory_bytes"], bool)
        or not isinstance(device["total_global_memory_bytes"], int)
        or device["wave_size"] != 32
        or device["total_global_memory_bytes"] <= 0
    ):
        raise ContractError("HIP wave size or device-local memory fact is invalid")
    if any(device[field] != 0 for field in ("allocation_count", "copy_count", "kernel_dispatch_count", "dispatch_count")):
        raise ContractError("HIP observation provider attempted allocation, copy, kernel, or dispatch")
    runtime = preflight["runtime"]
    expected_runtime = {
        "rocm_root": row["rocm"]["root"],
        "release": row["rocm"]["release"],
        "hip_runtime_api_version": row["rocm"]["hip_runtime_api_version"],
        "hip_runtime_library_path": row["rocm"]["hip_runtime_library"],
        "hsa_runtime_library_path": row["rocm"]["hsa_runtime_library"],
    }
    if any(runtime[key] != expected for key, expected in expected_runtime.items()):
        raise ContractError("runtime ROCm root/release/API/library path does not match the fixed tuple")
    if {
        "bdf": runtime["bdf"],
        "uuid": runtime["uuid"],
        "gcnArchName": runtime["gcnArchName"],
    } != {"bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"]}:
        raise ContractError("runtime observation is not bound to the canonical device")
    pre_health_time = parse_time(preflight["health_pre"]["observed_at"])
    post_health_time = parse_time(preflight["health_post"]["observed_at"])
    pre_process_time = parse_time(preflight["process_pre"]["observed_at"])
    post_process_time = parse_time(preflight["process_post"]["observed_at"])
    if pre_health_time >= post_health_time or pre_process_time >= post_process_time:
        raise ContractError("G0 pre/post observations are not ordered")
    if max(pre_health_time, pre_process_time) >= min(post_health_time, post_process_time):
        raise ContractError("G0 pre observations do not all precede post observations")
    validate_health(preflight["health_pre"], "pre", row)
    validate_health(preflight["health_post"], "post", row)
    for counter in ("device_state", "amdgpu_driver_bound", "runtime_status", "ras_uncorrectable_count", "sysfs_ras_uncorrectable_count"):
        if preflight["health_post"]["facts"][counter] != preflight["health_pre"]["facts"][counter]:
            raise ContractError(f"post health counter changed during identity-only preflight: {counter}")
    validate_processes(preflight["process_pre"], "pre", row)
    validate_processes(preflight["process_post"], "post", row)
    if preflight["scope"] != {
        "selected_backend": "hip-preflight",
        "fallback_allowed": False,
        "fallback_used": False,
        "identity_probe_only": True,
        "native_hip_observation_provider": "native-hip-observer-v1",
        "execution_verified": False,
        "numerics_verified": False,
        "performance_verified": False,
        "support_claim": False,
    }:
        raise ContractError("G0 scope overclaims execution, correctness, performance, fallback, or support")


def validate_observation_window(
    preflight: Mapping[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Require every successful G0 observation to be inside the report window."""

    if started_at > finished_at or finished_at > datetime.now(timezone.utc):
        raise ContractError("G0 report execution window is reversed or in the future")
    for field in ("health_pre", "health_post", "process_pre", "process_post"):
        observed_at = parse_time(preflight[field]["observed_at"])
        if not started_at <= observed_at <= finished_at:
            raise ContractError(f"G0 {field}.observed_at is outside the report execution window")


def validate_g0_preflight(
    preflight: dict[str, Any],
    row_id: str,
    repo: Path = ROOT,
    *,
    expected_sha: str | None = None,
    expected_tree: str | None = None,
    observation_window: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    schema, matrix = load_g0_contract(repo)
    validate_g0_matrix(repo)
    validate_schema(preflight, preflight_schema(schema), "G0 preflight")
    row = row_by_id(matrix, row_id)
    validate_candidate(preflight["candidate"], expected_sha=expected_sha, expected_tree=expected_tree)
    normalized = validate_visibility_environment(preflight["visibility"])
    if normalized != preflight["visibility"]:
        raise ContractError("visibility record is not canonical")
    validate_routing(preflight["routing"], preflight["visibility"], row)
    validate_artifact_binding(preflight["artifact_binding"], row, repo, preflight["candidate"])
    validate_observations(preflight, row, repo)
    if observation_window is not None:
        validate_observation_window(preflight, *observation_window)
    return row


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--preflight", type=Path)
    result.add_argument("--row", choices=("g0-gfx1030", "g0-gfx1201"))
    result.add_argument("--expected-candidate-sha")
    result.add_argument("--expected-tree-oid")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_g0_matrix(args.repo)
        if args.preflight is not None:
            if args.row is None:
                raise ContractError("--row is required with --preflight")
            document = read_json(args.preflight)
            if not isinstance(document, dict):
                raise ContractError("G0 preflight document must be an object")
            validate_g0_preflight(
                document,
                args.row,
                args.repo,
                expected_sha=args.expected_candidate_sha,
                expected_tree=args.expected_tree_oid,
            )
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"G0 contract validation: FAIL: {exc}", file=sys.stderr)
        return 1
    print("G0 contract validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
