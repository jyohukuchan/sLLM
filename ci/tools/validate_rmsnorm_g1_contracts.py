#!/usr/bin/env python3
"""Fail-closed contracts for controller-owned semantic RMSNorm G1 evidence.

Semantic G1 deliberately has no mutable Python authority or filesystem-to-PASS
API.  The controller keeps immutable descriptor snapshots and raw worker
frames live until it has recomputed the result itself.
"""

from __future__ import annotations

import argparse
import array
import base64
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import socket
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import ContractError, ROOT, exact_sha  # noqa: E402
from exact_actions import ExactActionError, validate_manifest as validate_exact_action_manifest  # noqa: E402


TARGETS = ("gfx1030", "gfx1201")
ROWS = tuple(f"rmsnorm-semantic-g1-{target}" for target in TARGETS)
MATRIX_SUITE_ID = "g1-rmsnorm-semantic-runtime"
EXPECTED_BINDINGS = {
    "gfx1030": {
        "bdf": "0000:03:00.0",
        "uuid": "GPU-76a08c022586fed6",
        "product": "AMD Radeon Pro V620",
        "physical_hip_index": 1,
        "logical_device_index": 0,
        "seed": 1031,
    },
    "gfx1201": {
        "bdf": "0000:07:00.0",
        "uuid": "GPU-a8e9ddefa2d60f55",
        "product": "AMD Radeon AI PRO R9700",
        "physical_hip_index": 2,
        "logical_device_index": 0,
        "seed": 1202,
    },
}
EXPECTED_SCOPE = {
    "selected_backend": "hip",
    "fallback_allowed": False,
    "fallback_used": False,
    "model_used": False,
    "semantic_op_used": True,
    "cpu_fallback_used": False,
    "gpu_execution": True,
    "private_g1_used": False,
    "h3_artifact_used": False,
}
EXPECTED_CASES = [
    {"id": f"r{rows}-n{width}", "rows": rows, "n": width, "classification": "finite", "nonfinite_input": "none"}
    for rows, width in ((1, 1), (1, 3), (2, 17), (1, 255), (1, 256), (1, 257), (1, 4095), (1, 4096))
]
EXPECTED_CASES.extend(
    [
        {"id": "r1-n2560", "rows": 1, "n": 2560, "classification": "finite", "nonfinite_input": "none"},
        {"id": "r1-n2560-nan", "rows": 1, "n": 2560, "classification": "nan", "nonfinite_input": "activation"},
        {"id": "r1-n2560-posinf", "rows": 1, "n": 2560, "classification": "posinf", "nonfinite_input": "activation"},
        {"id": "r1-n2560-neginf", "rows": 1, "n": 2560, "classification": "neginf", "nonfinite_input": "activation"},
        {"id": "r1-n2560-raw-scale-nan", "rows": 1, "n": 2560, "classification": "nan", "nonfinite_input": "raw_scale"},
        {"id": "r1-n2560-raw-scale-posinf", "rows": 1, "n": 2560, "classification": "posinf", "nonfinite_input": "raw_scale"},
        {"id": "r1-n2560-raw-scale-neginf", "rows": 1, "n": 2560, "classification": "neginf", "nonfinite_input": "raw_scale"},
    ]
)
EXPECTED_COMMAND = [
    "/usr/bin/cargo",
    "+1.97.1",
    "build",
    "--locked",
    "--offline",
    "--release",
    "--package",
    "sllm-hip",
    "--bin",
    "sllm-rmsnorm-g1-evidence",
]
EXPECTED_CASE_RESOURCE_COUNTS = {
    "allocation_count": 3,
    "copy_count": 3,
    "dispatch_count": 1,
    "kernel_count": 1,
}
EXPECTED_ROW_RESOURCE_COUNTS = {
    name: value * len(EXPECTED_CASES)
    for name, value in EXPECTED_CASE_RESOURCE_COUNTS.items()
}
EXPECTED_CODEGEN = {
    "gfx1030": {
        "target": "gfx1030",
        "target_kind": "exact",
        "target_count": 1,
        "code_object_version": "V6",
        "wavefront_size": 32,
        "features": {"xnack": "unsupported", "sramecc": "unsupported", "generic_processor_version": 0},
        "e_flags": "0x00000036",
    },
    "gfx1201": {
        "target": "gfx1201",
        "target_kind": "exact",
        "target_count": 1,
        "code_object_version": "V6",
        "wavefront_size": 32,
        "features": {"xnack": "unsupported", "sramecc": "unsupported", "generic_processor_version": 0},
        "e_flags": "0x0000004e",
    },
}

MATRIX_MANIFEST = "ci/matrix/rmsnorm-semantic-g1-v1.json"
MODEL_LOCK = "docs/models/locks/qwen3.5-4b-bf16.json"
MATRIX_SCHEMA = "ci/schema/rmsnorm-semantic-g1-matrix-v1.schema.json"
ARTIFACT_SCHEMA = "ci/schema/rmsnorm-semantic-g1-artifact-v1.schema.json"
REPORT_SCHEMA = "ci/schema/rmsnorm-semantic-g1-report-v1.schema.json"
AGGREGATE_SCHEMA = "ci/schema/rmsnorm-semantic-g1-aggregate-v1.schema.json"
WORKFLOW_PATH = ".github/workflows/semantic-rmsnorm-g1.yml"
SEMANTIC_G1_WORKFLOW_SHA256 = "a1c0cc85334445c14c15b5be43e979f587a4f2bd8cb8b53690603b65939770fc"
_SEMANTIC_G1_UPLOAD_ROOT = "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}"
SEMANTIC_G1_RESPONSE_UPLOAD_PATHS = tuple(
    f"{_SEMANTIC_G1_UPLOAD_ROOT}/rows/{row_id}/raw/case-{order}{suffix}"
    for row_id in ROWS
    for order in range(len(EXPECTED_CASES))
    for suffix in (".bin", ".bin.sha256")
)
SEMANTIC_G1_UPLOAD_PATHS = (
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rmsnorm-semantic-g1-aggregate.json",
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rmsnorm-semantic-g1-aggregate.json.sha256",
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rows/rmsnorm-semantic-g1-gfx1030/rmsnorm-semantic-g1-report.json",
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rows/rmsnorm-semantic-g1-gfx1030/rmsnorm-semantic-g1-report.json.sha256",
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rows/rmsnorm-semantic-g1-gfx1201/rmsnorm-semantic-g1-report.json",
    "${{ env.RUN_ROOT }}/rmsnorm-semantic-g1-aggregate-${{ github.run_id }}-${{ github.run_attempt }}/rows/rmsnorm-semantic-g1-gfx1201/rmsnorm-semantic-g1-report.json.sha256",
    "${{ env.RUN_ROOT }}/artifacts/rmsnorm-semantic-g1-gfx1030/rmsnorm-semantic-g1-artifact.json",
    "${{ env.RUN_ROOT }}/artifacts/rmsnorm-semantic-g1-gfx1030/rmsnorm-semantic-g1-artifact.json.sha256",
    "${{ env.RUN_ROOT }}/artifacts/rmsnorm-semantic-g1-gfx1201/rmsnorm-semantic-g1-artifact.json",
    "${{ env.RUN_ROOT }}/artifacts/rmsnorm-semantic-g1-gfx1201/rmsnorm-semantic-g1-artifact.json.sha256",
    *SEMANTIC_G1_RESPONSE_UPLOAD_PATHS,
)
SEMANTIC_G1_UPLOAD_PATH_TEXT = "\n".join(SEMANTIC_G1_UPLOAD_PATHS) + "\n"
RUNNER_RELATIVE_PATH = "ci/tools/run_rmsnorm_g1_runtime.py"
RUST_EVIDENCE_RELATIVE_PATH = "crates/sllm-hip/src/bin/sllm-rmsnorm-g1-evidence.rs"
RUST_BUILD_RELATIVE_PATH = "crates/sllm-hip-sys/build.rs"
HIP_CMAKE_RELATIVE_PATH = "native/hip/CMakeLists.txt"
METADATA_NAME = "rmsnorm-semantic-g1-artifact.json"
REPORT_NAME = "rmsnorm-semantic-g1-report.json"
BINARY_NAME = "sllm-rmsnorm-g1-evidence"
COMPANION_NAME = "device-code-object-{target}.elf"
SIDECAR_SUFFIX = ".sha256"
LOGICAL_KERNEL = "rmsnorm.baseline.wave32.v1"
DEVICE_SYMBOL = "sllm_rmsnorm_baseline_wave32_v1"
COMPILER_LOGICAL_PATH = "/opt/rocm/bin/amdclang++"
COMPILER_SNAPSHOT_SHA256 = "2ec8efcf34ee0676977e497e9611bf885927b8ef94922ec3ab5d39db926fa72b"
COMPILER_SNAPSHOT_SIZE = 19_720
COMPILER_TRANSCRIPT_MAX_BYTES = 64 * 1024 * 1024
EXACT_ACTION_PROTOCOL = "parent-issued-exact-action-v1"
EXACT_ACTION_MANIFEST_VERSION = "exact-parent-action-manifest-v1"
CONTROLLER_PYTHON_RECORD = {
    "path": "/usr/bin/python3",
    "resolved_path": "/usr/bin/python3.12",
    "size_bytes": 8_020_928,
    "sha256": "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
}
COMPILER_CLIENT_INTERPRETER_RECORD = CONTROLLER_PYTHON_RECORD
COMPILER_SOURCE_RECORD = {
    "path": COMPILER_LOGICAL_PATH,
    "resolved_path": "/opt/rocm/core-7.14/lib/llvm/bin/amdllvm",
    "size_bytes": COMPILER_SNAPSHOT_SIZE,
    "sha256": COMPILER_SNAPSHOT_SHA256,
}
AUTHORITY_VERSION = "rmsnorm-semantic-g1-reviewed-authority-v1"
AUTHORITY_SOURCE_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/sllm-core/Cargo.toml",
    "crates/sllm-core/src/backend.rs",
    "crates/sllm-core/src/dtype.rs",
    "crates/sllm-core/src/execution.rs",
    "crates/sllm-core/src/fake.rs",
    "crates/sllm-core/src/handles.rs",
    "crates/sllm-core/src/lib.rs",
    "crates/sllm-core/src/model.rs",
    "crates/sllm-core/src/op.rs",
    "crates/sllm-core/src/registry.rs",
    "crates/sllm-core/src/tensor.rs",
    "crates/sllm-hip-sys/Cargo.toml",
    "crates/sllm-hip-sys/build.rs",
    "crates/sllm-hip-sys/src/bindings.rs",
    "crates/sllm-hip-sys/src/evidence_bindings.rs",
    "crates/sllm-hip-sys/src/lib.rs",
    "crates/sllm-hip/Cargo.toml",
    "crates/sllm-hip/src/bin/sllm-hip-evidence.rs",
    "crates/sllm-hip/src/lib.rs",
    "crates/sllm-hip/src/bridge.rs",
    "crates/sllm-hip/src/rmsnorm.rs",
    "crates/sllm-hip/src/runtime.rs",
    RUST_EVIDENCE_RELATIVE_PATH,
    "include/sllm/hip.h",
    "include/sllm/sllm.h",
    "native/hip/CMakeLists.txt",
    "native/hip/src/abi_layout_probe.cpp",
    "native/hip/src/evidence_abi.h",
    "native/hip/src/header_c_compile.c",
    "native/hip/src/header_cpp_compile.cpp",
    "native/hip/src/hip_compile_probe.hip.cpp",
    "native/hip/src/hip_evidence_runtime.hip.cpp",
    "native/hip/src/hip_evidence_stub.cpp",
    "native/hip/src/hip_stub.cpp",
    "native/hip/src/public_runtime.hip.cpp",
    "native/hip/src/public_runtime_internal.hpp",
    "native/hip/src/public_runtime_stub.cpp",
    "native/hip/src/rmsnorm_api.cpp",
    "native/hip/src/rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_kernel.hip.cpp",
    "native/hip/src/rmsnorm_kernel_internal.hpp",
    MODEL_LOCK,
    "ci/tools/orchestrate_rmsnorm_g1_evidence.py",
    "ci/tools/common.py",
    "ci/tools/exact_actions.py",
    "ci/tools/validate_rmsnorm_g1_contracts.py",
    "ci/tools/build_rmsnorm_g1_runtime.py",
    "ci/tools/run_rmsnorm_g1_runtime.py",
    "ci/tools/validate_g0_contracts.py",
    "ci/tools/run_g0_preflight.py",
    "ci/tools/validate_h3_contracts.py",
    MATRIX_MANIFEST,
    MATRIX_SCHEMA,
    ARTIFACT_SCHEMA,
    REPORT_SCHEMA,
    AGGREGATE_SCHEMA,
    WORKFLOW_PATH,
)
# These are the non-code inputs that decide the semantic G1 row shape and the
# admissibility of every emitted artifact/report/aggregate.  During a real
# controller run they are read from the controller's already-verified Git
# object byte map, rather than reopened through a mutable checkout pathname.
REVIEWED_CONTRACT_FILES = (
    MATRIX_MANIFEST,
    MATRIX_SCHEMA,
    ARTIFACT_SCHEMA,
    REPORT_SCHEMA,
    AGGREGATE_SCHEMA,
    WORKFLOW_PATH,
    MODEL_LOCK,
)

# This is the reviewed semantic-G1 ABI/dispatch closure.  The public ABI,
# native kernel/runtime sources, and Rust dispatch adapters are deliberately
# listed by role here; the validator below intersects this list with the
# immutable authority closure instead of allowing a broad glob to imply G1
# ownership.
SEMANTIC_G1_DISPATCH_AUTHORITY_INPUTS = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/sllm-core/Cargo.toml",
    "crates/sllm-core/src/backend.rs",
    "crates/sllm-core/src/dtype.rs",
    "crates/sllm-core/src/execution.rs",
    "crates/sllm-core/src/handles.rs",
    "crates/sllm-core/src/lib.rs",
    "crates/sllm-core/src/op.rs",
    "crates/sllm-core/src/registry.rs",
    "crates/sllm-core/src/tensor.rs",
    "crates/sllm-hip-sys/Cargo.toml",
    "crates/sllm-hip-sys/build.rs",
    "crates/sllm-hip-sys/src/bindings.rs",
    "crates/sllm-hip-sys/src/evidence_bindings.rs",
    "crates/sllm-hip-sys/src/lib.rs",
    "crates/sllm-hip/Cargo.toml",
    "crates/sllm-hip/src/bin/sllm-rmsnorm-g1-evidence.rs",
    "crates/sllm-hip/src/lib.rs",
    "crates/sllm-hip/src/bridge.rs",
    "crates/sllm-hip/src/rmsnorm.rs",
    "crates/sllm-hip/src/runtime.rs",
    "include/sllm/hip.h",
    "include/sllm/sllm.h",
    "native/hip/CMakeLists.txt",
    "native/hip/src/hip_stub.cpp",
    "native/hip/src/evidence_abi.h",
    "native/hip/src/hip_evidence_stub.cpp",
    "native/hip/src/header_c_compile.c",
    "native/hip/src/header_cpp_compile.cpp",
    "native/hip/src/public_runtime.hip.cpp",
    "native/hip/src/public_runtime_internal.hpp",
    "native/hip/src/rmsnorm_api.cpp",
    "native/hip/src/rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_kernel.hip.cpp",
    "native/hip/src/rmsnorm_kernel_internal.hpp",
)


def semantic_g1_required_path_ownership() -> tuple[str, ...]:
    authority = set(AUTHORITY_SOURCE_FILES)
    missing = [path for path in SEMANTIC_G1_DISPATCH_AUTHORITY_INPUTS if path not in authority]
    if missing:
        raise EvidenceError(f"semantic G1 dispatch closure is not covered by reviewed authority: {missing}")
    return tuple(path for path in AUTHORITY_SOURCE_FILES if path in authority and path in SEMANTIC_G1_DISPATCH_AUTHORITY_INPUTS)


_BOUND_REVIEWED_CONTRACT_REPOSITORY: Path | None = None
_BOUND_REVIEWED_CONTRACT_BYTES: dict[str, bytes] | None = None
MAX_OUTPUT = 1024 * 1024
MAX_IPC_FRAME = MAX_OUTPUT
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
GIT_OBJECT_FORMATS = {"sha1": 40, "sha256": 64}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SANITIZED_RUNTIME_PATH = "/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SANITIZED_RUNTIME_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib"
CANONICAL_PYTHON = Path("/usr/bin/python3")
CANONICAL_CARGO = Path("/usr/bin/cargo")
CANONICAL_CMAKE = Path("/usr/bin/cmake")
CANONICAL_CXX = Path("/usr/bin/c++")
CANONICAL_RUSTUP_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".rustup"
CANONICAL_CARGO_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".cargo"
PROCESS_LIMITER_PATH = Path("/usr/bin/prlimit")
CANONICAL_TOOL_PATHS = {
    "cargo": CANONICAL_CARGO,
    "rustc": CANONICAL_RUSTUP_HOME / "toolchains/1.97.1-x86_64-unknown-linux-gnu/bin/rustc",
    "cmake": CANONICAL_CMAKE,
    "cxx": CANONICAL_CXX,
    "objcopy": Path("/opt/rocm/lib/llvm/bin/llvm-objcopy"),
    "bundler": Path("/opt/rocm/lib/llvm/bin/clang-offload-bundler"),
    "amd_smi": Path("/opt/rocm/core-7.14/bin/amd-smi"),
    "process_limiter": PROCESS_LIMITER_PATH,
    "loader": Path("/lib64/ld-linux-x86-64.so.2"),
    "hip_runtime": Path("/opt/rocm/lib/libamdhip64.so"),
    "hsa_runtime": Path("/opt/rocm/lib/libhsa-runtime64.so"),
}
CANONICAL_TOOL_KEYS = tuple(CANONICAL_TOOL_PATHS)
MODEL_HIDDEN_SIZE = 2560
MODEL_EPSILON = 1e-6
MODEL_EPSILON_TEXT = "1e-6"
MODEL_LOCK_SHA256 = "e0ab289154c0b59c8dc5863fd14024a3228f6ea20571c12a805662a090e61abb"
REQUIRED_SEALS = (
    getattr(fcntl, "F_SEAL_SHRINK", 0)
    | getattr(fcntl, "F_SEAL_GROW", 0)
    | getattr(fcntl, "F_SEAL_WRITE", 0)
    | getattr(fcntl, "F_SEAL_SEAL", 0)
)


def _validate_exact_action(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Translate reusable exact-action validation into the G1 error domain."""

    try:
        checked = validate_exact_action_manifest(manifest)
    except ExactActionError as exc:
        raise EvidenceError(f"semantic G1 exact action manifest is invalid: {exc}") from exc
    if checked["schema_version"] != EXACT_ACTION_MANIFEST_VERSION or checked["target"] not in TARGETS:
        raise EvidenceError("semantic G1 exact action manifest version/target is not canonical")
    return checked


class EvidenceError(ContractError):
    """A malformed or unauthorised semantic-G1 evidence input."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("semantic G1 value is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError(f"cannot parse strict {label} JSON") from exc


def _open_regular(path: Path, label: str) -> int:
    if not isinstance(path, Path) or not path.is_absolute() or "\x00" in str(path):
        raise EvidenceError(f"{label} must be an absolute path")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {label}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size < 1:
            raise EvidenceError(f"{label} is not a non-empty regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def fd_read_all(descriptor: int, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    try:
        size = os.fstat(descriptor).st_size
    except OSError as exc:
        raise EvidenceError("cannot inspect descriptor") from exc
    if size < 0 or size > max_bytes:
        raise EvidenceError("descriptor exceeds bounded read limit")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise EvidenceError("descriptor changed during bounded read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def fd_sha256(descriptor: int) -> str:
    try:
        size = os.fstat(descriptor).st_size
    except OSError as exc:
        raise EvidenceError("cannot hash closed descriptor") from exc
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise EvidenceError("descriptor changed while hashing")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def fd_path(descriptor: int) -> Path:
    if not isinstance(descriptor, int) or descriptor < 0:
        raise EvidenceError("invalid retained descriptor")
    return Path(f"/proc/self/fd/{descriptor}")


def _record_from_descriptor(descriptor: int, *, path: Path) -> dict[str, Any]:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_size < 1:
        raise EvidenceError("descriptor is not a non-empty regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("descriptor source cannot be resolved") from exc
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": details.st_size,
        "sha256": fd_sha256(descriptor),
    }


def file_identity(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"{label} source path cannot be resolved") from exc
    descriptor = _open_regular(resolved, label)
    try:
        record = _record_from_descriptor(descriptor, path=path)
        if record["resolved_path"] != str(resolved):
            raise EvidenceError(f"{label} source path changed while recording identity")
        return record
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return str(file_identity(path, "hash input")["sha256"])


def _validate_digest_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {"path", "resolved_path", "size_bytes", "sha256"}
    if set(record) != required:
        raise EvidenceError(f"{label} descriptor record is not closed")
    if not isinstance(record["path"], str) or not isinstance(record["resolved_path"], str):
        raise EvidenceError(f"{label} descriptor path is malformed")
    path = Path(record["path"])
    resolved = Path(record["resolved_path"])
    if not path.is_absolute() or not resolved.is_absolute() or "\x00" in str(path) or "\x00" in str(resolved):
        raise EvidenceError(f"{label} descriptor path is unsafe")
    if isinstance(record["size_bytes"], bool) or not isinstance(record["size_bytes"], int) or record["size_bytes"] < 1:
        raise EvidenceError(f"{label} descriptor size is malformed")
    if not isinstance(record["sha256"], str) or SHA256_RE.fullmatch(record["sha256"]) is None:
        raise EvidenceError(f"{label} descriptor hash is malformed")
    return dict(record)


def _validate_sealed_input_view(view: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    required = {"algorithm", "argv", "argv_sha256", "inputs", "include_directories", "sealed"}
    if not isinstance(view, Mapping) or set(view) != required or view.get("algorithm") != "sealed-input-view-v1" or view.get("sealed") is not True:
        raise EvidenceError("exact-action compiler input view is not the reviewed sealed form")
    argv = view.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value or "\0" in value for value in argv) or view.get("argv_sha256") != sha256_json(argv):
        raise EvidenceError("exact-action sealed input view argv digest is malformed")
    inputs = view.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(manifest["inputs"]):
        raise EvidenceError("exact-action sealed input view omitted an issued input")
    target_fds: set[int] = set()
    for source, actual in zip(manifest["inputs"], inputs, strict=True):
        expected = dict(source)
        view_fd = actual.get("view_fd") if isinstance(actual, Mapping) else None
        if not isinstance(view_fd, int) or isinstance(view_fd, bool) or view_fd < 300 or view_fd in target_fds or set(actual) != set(expected) | {"view_fd"} or any(actual.get(key) != value for key, value in expected.items()):
            raise EvidenceError("exact-action sealed input view identity is not bound to the issued input")
        target_fds.add(view_fd)
    include_directories = view.get("include_directories")
    if not isinstance(include_directories, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"original", "view"}
        or not isinstance(item.get("original"), str)
        or not isinstance(item.get("view"), str)
        or not Path(item["view"]).is_absolute()
        or not item["view"].startswith("/tmp/sllm-exact-input-view-")
        for item in include_directories
    ):
        raise EvidenceError("exact-action sealed include directory view is malformed")
    original_inputs = {str(record["path"]) for record in manifest["inputs"]}
    if any(value in original_inputs for value in argv):
        raise EvidenceError("exact-action compiler argv still contains a mutable input pathname")
    return list(argv)


def validate_open_file_identity(descriptor: int, expected: Mapping[str, Any], label: str) -> None:
    expected_record = _validate_digest_record(expected, label)
    details = os.fstat(descriptor)
    if details.st_size != expected_record["size_bytes"] or fd_sha256(descriptor) != expected_record["sha256"]:
        raise EvidenceError(f"{label} descriptor bytes do not match its immutable record")


def _sealed_memfd(data: bytes, label: str) -> int:
    if not data:
        raise EvidenceError(f"{label} cannot be empty")
    if not hasattr(os, "memfd_create") or not hasattr(fcntl, "F_ADD_SEALS"):
        raise EvidenceError("Linux sealed memfd support is required for semantic G1")
    flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
    descriptor = -1
    try:
        descriptor = os.memfd_create(f"sllm-semantic-g1-{label}", flags)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise EvidenceError(f"cannot create sealed {label} descriptor") from exc


def descriptor_is_sealed(descriptor: int) -> bool:
    try:
        return (fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & REQUIRED_SEALS) == REQUIRED_SEALS
    except OSError:
        return False


@dataclass
class SealedDescriptor:
    """An immutable byte snapshot retained by the controller until completion."""

    fd: int
    record: dict[str, Any]
    label: str

    def close(self) -> None:
        if self.fd >= 0:
            try:
                os.close(self.fd)
            finally:
                self.fd = -1


def snapshot_file(path: Path, expected: Mapping[str, Any] | None, label: str) -> SealedDescriptor:
    """Read and verify a file once, then retain only a sealed byte snapshot."""

    expected_record = None if expected is None else _validate_digest_record(expected, label)
    source_path = path
    if expected_record is not None:
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            raise EvidenceError(f"{label} source path cannot be resolved") from exc
        if str(resolved_path) != expected_record["resolved_path"]:
            raise EvidenceError(f"{label} source path resolved identity differs from the expected record")
        # A reviewed PT_INTERP may be a stable system symlink.  Open the
        # already-bound resolved target, never the mutable symlink pathname.
        source_path = Path(expected_record["resolved_path"])
    source = _open_regular(source_path, label)
    try:
        source_record = _record_from_descriptor(source, path=path)
        if expected_record is not None:
            if source_record != expected_record:
                raise EvidenceError(f"{label} changed before its immutable snapshot")
        data = fd_read_all(source, max_bytes=max(MAX_OUTPUT * 64, source_record["size_bytes"]))
    finally:
        os.close(source)
    descriptor = _sealed_memfd(data, label)
    record = {**source_record, "size_bytes": len(data), "sha256": sha256_bytes(data)}
    if expected_record is not None and record != expected_record:
        os.close(descriptor)
        raise EvidenceError(f"{label} immutable copy does not match expected bytes")
    return SealedDescriptor(descriptor, record, label)


def snapshot_bytes(data: bytes, *, logical_path: str, label: str) -> SealedDescriptor:
    descriptor = _sealed_memfd(data, label)
    return SealedDescriptor(descriptor, {"path": logical_path, "resolved_path": logical_path, "size_bytes": len(data), "sha256": sha256_bytes(data)}, label)


def _sidecar_text(digest: str, name: str) -> bytes:
    return f"{digest}  {name}\n".encode("ascii")


def validate_sidecar(descriptor: SealedDescriptor, *, target_record: Mapping[str, Any], filename: str, label: str) -> None:
    validate_open_file_identity(descriptor.fd, descriptor.record, label)
    if fd_read_all(descriptor.fd, max_bytes=1024) != _sidecar_text(str(target_record["sha256"]), filename):
        raise EvidenceError(f"{label} is not the immutable digest sidecar for its target")


def read_json(path: Path) -> Any:
    descriptor = _open_regular(path, "JSON input")
    try:
        return read_json_bytes(fd_read_all(descriptor, max_bytes=MAX_OUTPUT * 4), str(path))
    finally:
        os.close(descriptor)


def bind_controller_reviewed_sources(repo: Path, sources: Mapping[str, bytes]) -> None:
    """Bind real-controller contract reads to its sealed Git-object bytes.

    The controller calls this only after its fresh-process gate has checked
    the full closed source set against the reviewed commit/tree.  Host-side
    imports can use the normal filesystem validators, but they cannot turn a
    supplied map into a controller or evidence-emission capability.
    """

    repository = canonical_repository(repo)
    if set(sources) != set(AUTHORITY_SOURCE_FILES):
        raise EvidenceError("controller reviewed source map is not the exact closed authority source set")
    bound: dict[str, bytes] = {}
    for relative in REVIEWED_CONTRACT_FILES:
        value = sources.get(relative)
        if not isinstance(value, bytes) or not value:
            raise EvidenceError("controller reviewed matrix/schema/workflow bytes are unavailable")
        bound[relative] = bytes(value)
    global _BOUND_REVIEWED_CONTRACT_REPOSITORY, _BOUND_REVIEWED_CONTRACT_BYTES
    if _BOUND_REVIEWED_CONTRACT_BYTES is not None:
        if _BOUND_REVIEWED_CONTRACT_REPOSITORY != repository or _BOUND_REVIEWED_CONTRACT_BYTES != bound:
            raise EvidenceError("controller reviewed contract bytes cannot be rebound")
        return
    _BOUND_REVIEWED_CONTRACT_REPOSITORY = repository
    _BOUND_REVIEWED_CONTRACT_BYTES = bound


def _reviewed_contract_bytes(repo: Path, relative: str) -> bytes | None:
    """Return controller-held immutable contract bytes when execution bound them."""

    if _BOUND_REVIEWED_CONTRACT_BYTES is None:
        return None
    if _BOUND_REVIEWED_CONTRACT_REPOSITORY != repo:
        raise EvidenceError("controller contract read uses a repository outside the reviewed workspace")
    value = _BOUND_REVIEWED_CONTRACT_BYTES.get(relative)
    if value is None:
        raise EvidenceError("controller contract read is outside the closed reviewed source map")
    return value


_CLOSED_SCHEMA_KEYWORDS = {
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "allOf",
    "const",
    "enum",
    "format",
    "items",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "oneOf",
    "pattern",
    "prefixItems",
    "patternProperties",
    "properties",
    "required",
    "uniqueItems",
    "type",
}
_CLOSED_SCHEMA_TYPES = {"array", "boolean", "integer", "number", "object", "string"}


def _check_closed_schema_node(node: Any, root: Mapping[str, Any], path: str) -> None:
    """Check the exact JSON-Schema subset accepted by the sealed controller."""

    if node is False:
        return
    if not isinstance(node, Mapping):
        raise EvidenceError(f"closed schema node is not an object at {path}")
    unknown = set(node) - _CLOSED_SCHEMA_KEYWORDS
    if unknown:
        raise EvidenceError(f"closed schema uses unsupported keywords at {path}: {sorted(unknown)}")
    if "$ref" in node:
        reference = node["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/$defs/") or "/" in reference[8:]:
            raise EvidenceError(f"closed schema reference is not a local definition at {path}")
        definitions = root.get("$defs")
        if not isinstance(definitions, Mapping) or reference[8:] not in definitions:
            raise EvidenceError(f"closed schema reference is unresolved at {path}")
    if "type" in node and node["type"] not in _CLOSED_SCHEMA_TYPES:
        raise EvidenceError(f"closed schema type is unsupported at {path}")
    if "additionalProperties" in node and node["additionalProperties"] is not False:
        raise EvidenceError(f"closed schema additionalProperties must be false at {path}")
    if "properties" in node:
        properties = node["properties"]
        if not isinstance(properties, Mapping) or any(not isinstance(name, str) for name in properties):
            raise EvidenceError(f"closed schema properties are malformed at {path}")
        for name, child in properties.items():
            _check_closed_schema_node(child, root, f"{path}/properties/{name}")
    if "patternProperties" in node:
        patterns = node["patternProperties"]
        if not isinstance(patterns, Mapping):
            raise EvidenceError(f"closed schema patternProperties are malformed at {path}")
        for pattern, child in patterns.items():
            if not isinstance(pattern, str):
                raise EvidenceError(f"closed schema patternProperties key is malformed at {path}")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise EvidenceError(f"closed schema patternProperties regex is invalid at {path}") from exc
            _check_closed_schema_node(child, root, f"{path}/patternProperties/{pattern}")
    if "$defs" in node:
        definitions = node["$defs"]
        if not isinstance(definitions, Mapping) or any(not isinstance(name, str) or not name for name in definitions):
            raise EvidenceError(f"closed schema definitions are malformed at {path}")
        for name, child in definitions.items():
            _check_closed_schema_node(child, root, f"{path}/$defs/{name}")
    if node.get("type") == "object" and node.get("additionalProperties") is not False:
        raise EvidenceError(f"closed schema object must reject additional properties at {path}")
    if "required" in node:
        required = node["required"]
        if not isinstance(required, list) or any(not isinstance(name, str) for name in required) or len(set(required)) != len(required):
            raise EvidenceError(f"closed schema required list is malformed at {path}")
    if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
        raise EvidenceError(f"closed schema enum is malformed at {path}")
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise EvidenceError(f"closed schema uniqueItems is malformed at {path}")
    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            raise EvidenceError(f"closed schema pattern is malformed at {path}")
        try:
            re.compile(node["pattern"])
        except re.error as exc:
            raise EvidenceError(f"closed schema pattern is invalid at {path}") from exc
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        if keyword in node and (isinstance(node[keyword], bool) or not isinstance(node[keyword], int) or node[keyword] < 0):
            raise EvidenceError(f"closed schema {keyword} is malformed at {path}")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (isinstance(node[keyword], bool) or not isinstance(node[keyword], (int, float))):
            raise EvidenceError(f"closed schema {keyword} is malformed at {path}")
    if "prefixItems" in node:
        if not isinstance(node["prefixItems"], list):
            raise EvidenceError(f"closed schema prefixItems are malformed at {path}")
        for index, child in enumerate(node["prefixItems"]):
            _check_closed_schema_node(child, root, f"{path}/prefixItems/{index}")
    if "items" in node:
        _check_closed_schema_node(node["items"], root, f"{path}/items")
    for keyword in ("allOf", "oneOf"):
        if keyword in node:
            branches = node[keyword]
            if not isinstance(branches, list) or not branches:
                raise EvidenceError(f"closed schema {keyword} is malformed at {path}")
            for index, child in enumerate(branches):
                _check_closed_schema_node(child, root, f"{path}/{keyword}/{index}")
    if "format" in node and node["format"] != "date-time":
        raise EvidenceError(f"closed schema format is unsupported at {path}")
    for keyword in ("$id", "$schema"):
        if keyword in node and not isinstance(node[keyword], str):
            raise EvidenceError(f"closed schema {keyword} is malformed at {path}")


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    return left == right


def _date_time_valid(value: Any) -> bool:
    """Use one RFC3339 date-time predicate for the host and sealed paths."""

    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})",
        value,
    ) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _format_checker() -> Any:
    from jsonschema import FormatChecker

    checker = FormatChecker()
    checker.checks("date-time")(_date_time_valid)
    return checker


def _closed_type_matches(document: Any, expected: str) -> bool:
    return {
        "array": isinstance(document, list),
        "boolean": isinstance(document, bool),
        "integer": isinstance(document, int) and not isinstance(document, bool),
        "number": isinstance(document, (int, float)) and not isinstance(document, bool),
        "object": isinstance(document, dict),
        "string": isinstance(document, str),
    }[expected]


def _closed_schema_errors(document: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if schema is True:
        return errors
    if schema is False:
        return [f"{path}: schema is false"]
    if "$ref" in schema:
        definition = root["$defs"][str(schema["$ref"])[8:]]
        errors.extend(_closed_schema_errors(document, definition, root, path))
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _closed_type_matches(document, expected_type):
        return [f"{path}: expected {expected_type}"]
    if "const" in schema and not _json_equal(document, schema["const"]):
        errors.append(f"{path}: value differs from const")
    if "enum" in schema and not any(_json_equal(document, value) for value in schema["enum"]):
        errors.append(f"{path}: value is outside enum")
    if isinstance(document, dict):
        properties = schema.get("properties", {})
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        patterns = schema.get("patternProperties", {})
        for name, value in document.items():
            if isinstance(properties, Mapping) and name in properties:
                errors.extend(_closed_schema_errors(value, properties[name], root, f"{path}/{name}"))
            matched_pattern = False
            if isinstance(patterns, Mapping):
                for pattern, child in patterns.items():
                    if re.search(str(pattern), name) is not None:
                        matched_pattern = True
                        errors.extend(_closed_schema_errors(value, child, root, f"{path}/{name}"))
            if schema.get("additionalProperties") is False and name not in property_names and not matched_pattern:
                errors.append(f"{path}/{name}: additional property")
        for name in schema.get("required", []):
            if name not in document:
                errors.append(f"{path}/{name}: required property is missing")
    if isinstance(document, list):
        if "minItems" in schema and len(document) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(document) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(prefix[: len(document)]):
            errors.extend(_closed_schema_errors(document[index], child, root, f"{path}/{index}"))
        items = schema.get("items")
        start = len(prefix)
        if items is False and len(document) > start:
            errors.append(f"{path}/{start}: items are forbidden after prefixItems")
        elif isinstance(items, Mapping):
            for index in range(start, len(document)):
                errors.extend(_closed_schema_errors(document[index], items, root, f"{path}/{index}"))
        if schema.get("uniqueItems"):
            for left in range(len(document)):
                if any(_json_equal(document[left], document[right]) for right in range(left)):
                    errors.append(f"{path}: items are not unique")
                    break
    if isinstance(document, str):
        if "minLength" in schema and len(document) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength")
        if "maxLength" in schema and len(document) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], document) is None:
            errors.append(f"{path}: pattern mismatch")
        if schema.get("format") == "date-time":
            if not _date_time_valid(document):
                errors.append(f"{path}: invalid date-time")
    if isinstance(document, (int, float)) and not isinstance(document, bool):
        if "minimum" in schema and document < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and document > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    for child in schema.get("allOf", []):
        errors.extend(_closed_schema_errors(document, child, root, path))
    if "oneOf" in schema:
        matches = sum(not _closed_schema_errors(document, child, root, path) for child in schema["oneOf"])
        if matches != 1:
            errors.append(f"{path}: expected exactly one oneOf branch, got {matches}")
    return errors


def _schema(repo: Path, relative: str) -> dict[str, Any]:
    data = _reviewed_contract_bytes(repo, relative)
    document = read_json_bytes(data, relative) if data is not None else read_json(repo / relative)
    if not isinstance(document, dict):
        raise EvidenceError(f"schema is not an object: {relative}")
    if data is not None:
        _check_closed_schema_node(document, document, "<root>")
    else:
        try:
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(document)
        except Exception as exc:
            raise EvidenceError(f"invalid Draft 2020-12 schema: {relative}") from exc
    return document


def validate_schema(document: Any, schema: Mapping[str, Any], label: str) -> None:
    closed_errors = _closed_schema_errors(document, schema, schema, "<root>")
    try:
        from jsonschema import Draft202012Validator, FormatChecker

        external_errors = sorted(
            Draft202012Validator(schema, format_checker=_format_checker()).iter_errors(document),
            key=lambda error: list(error.absolute_path),
        )
    except ImportError:
        if closed_errors:
            raise EvidenceError(f"{label} violates its closed schema: {closed_errors[0]}")
        return
    if bool(closed_errors) != bool(external_errors):
        raise EvidenceError(f"{label} host/stdlib schema validators disagree")
    if closed_errors:
        error = external_errors[0] if external_errors else None
        if error is None:
            raise EvidenceError(f"{label} violates its closed schema: {closed_errors[0]}")
        where = "/".join(str(item) for item in error.absolute_path) or "<root>"
        raise EvidenceError(f"{label} violates its closed schema at {where}: {error.message}")


def _process_starttime(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return int(text.rsplit(") ", 1)[1].split()[19])
    except (OSError, IndexError, ValueError) as exc:
        raise EvidenceError("cannot read process starttime") from exc


def _process_real_ids(pid: int) -> tuple[int, int]:
    try:
        values: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith(("Uid:", "Gid:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.split()[0])
        return values["Uid"], values["Gid"]
    except (OSError, KeyError, ValueError, IndexError) as exc:
        raise EvidenceError("cannot read process credentials") from exc


def process_binding(pid: int) -> dict[str, int]:
    uid, gid = _process_real_ids(pid)
    return {"pid": pid, "starttime": _process_starttime(pid), "uid": uid, "gid": gid}


def verify_process_binding(binding: Mapping[str, Any]) -> None:
    if set(binding) != {"pid", "starttime", "uid", "gid"}:
        raise EvidenceError("process binding is not closed")
    if any(isinstance(binding[name], bool) or not isinstance(binding[name], int) for name in binding) or int(binding["pid"]) < 1:
        raise EvidenceError("process binding is invalid")
    if process_binding(int(binding["pid"])) != dict(binding):
        raise EvidenceError("process PID/starttime/UID/GID binding changed")


def controller_socketpair() -> tuple[socket.socket, socket.socket]:
    try:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        child.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        return parent, child
    except OSError as exc:
        raise EvidenceError("cannot create controller seqpacket channel") from exc


def _ipc_payload(document: Mapping[str, Any]) -> bytes:
    if not isinstance(document, Mapping):
        raise EvidenceError("controller frame is not an object")
    payload = canonical_bytes(dict(document))
    if not payload or len(payload) > MAX_IPC_FRAME:
        raise EvidenceError("controller frame exceeds its bounded size")
    return payload


def ipc_send(sock: socket.socket, document: Mapping[str, Any]) -> None:
    payload = _ipc_payload(document)
    try:
        sent = sock.send(payload)
    except OSError as exc:
        raise EvidenceError("cannot send controller frame") from exc
    if sent != len(payload):
        raise EvidenceError("controller frame short write")


def _close_rights(data: bytes) -> None:
    values = array.array("i")
    values.frombytes(data[: len(data) - (len(data) % values.itemsize)])
    for descriptor in set(values):
        try:
            os.close(descriptor)
        except OSError:
            pass


def ipc_recv(sock: socket.socket) -> tuple[dict[str, Any], tuple[int, int, int]]:
    ancillary = socket.CMSG_SPACE(struct.calcsize("3i")) + socket.CMSG_SPACE(16 * struct.calcsize("i"))
    try:
        payload, control, flags, _address = sock.recvmsg(MAX_IPC_FRAME + 1, ancillary)
    except OSError as exc:
        raise EvidenceError("cannot receive controller frame") from exc
    credentials: tuple[int, int, int] | None = None
    bad_ancillary = False
    for level, kind, data in control:
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
            if len(data) < struct.calcsize("3i") or credentials is not None:
                bad_ancillary = True
            else:
                credentials = struct.unpack("3i", data[: struct.calcsize("3i")])
        elif level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            _close_rights(data)
            bad_ancillary = True
        else:
            bad_ancillary = True
    # Walk all ancillary records before rejecting the frame.  In particular,
    # an unknown record before SCM_RIGHTS must not bypass descriptor cleanup;
    # MSG_CTRUNC is rejected after all delivered rights have been closed.
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not payload or len(payload) > MAX_IPC_FRAME:
        raise EvidenceError("controller frame is truncated or exceeds its bound")
    if bad_ancillary or credentials is None:
        raise EvidenceError("controller frame ancillary data is malformed or transfers descriptors")
    document = read_json_bytes(payload, "controller frame")
    if not isinstance(document, dict):
        raise EvidenceError("controller frame JSON is not an object")
    return document, credentials


def semantic_runtime_environment(physical_hip_index: int) -> dict[str, str]:
    if type(physical_hip_index) is not int or physical_hip_index < 0:
        raise EvidenceError("semantic G1 physical HIP index is invalid")
    return {
        "PATH": SANITIZED_RUNTIME_PATH,
        "LD_LIBRARY_PATH": SANITIZED_RUNTIME_LD_LIBRARY_PATH,
        "HIP_VISIBLE_DEVICES": str(physical_hip_index),
    }


def _identity(values: Mapping[str, Any], label: str) -> dict[str, str]:
    result = {name: str(values.get(name, "")) for name in ("reviewed_sha", "tested_sha", "workflow_sha")}
    for name, value in result.items():
        if GIT_OID_RE.fullmatch(value) is None:
            raise EvidenceError(f"{label}.{name} is not a supported Git object ID")
    if len({result["reviewed_sha"], result["tested_sha"], result["workflow_sha"]}) != 1:
        raise EvidenceError(f"{label} candidate SHAs differ")
    supplied_tree = values.get("git_tree_oid")
    if supplied_tree is not None and GIT_OID_RE.fullmatch(str(supplied_tree)) is None:
        raise EvidenceError(f"{label}.git_tree_oid is not a supported Git object ID")
    return result


def _candidate_document(identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    base = _identity(identity, label)
    for name in ("git_tree_oid", "git_object_format", "git_oid_width"):
        if name not in identity:
            raise EvidenceError(f"{label} is missing internally recomputed {name}")
    object_format = str(identity["git_object_format"])
    oid_width = identity["git_oid_width"]
    if object_format not in GIT_OBJECT_FORMATS or oid_width != GIT_OBJECT_FORMATS[object_format]:
        raise EvidenceError(f"{label} Git object format/width binding is invalid")
    if len(str(identity["git_tree_oid"])) != oid_width:
        raise EvidenceError(f"{label} Git tree OID width is invalid")
    return {
        **base,
        "git_tree_oid": str(identity["git_tree_oid"]),
        "git_object_format": object_format,
        "git_oid_width": oid_width,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }


def canonical_repository(repo: Path | None = None) -> Path:
    """Normalize a repository for validation-only helpers.

    This helper intentionally has no source-location based authority.  A
    validator may inspect a supplied checkout, but the controller obtains its
    authority exclusively from the immutable Git candidate and the closed CI
    workspace gate in :func:`verify_repository_identity`.
    """

    requested = ROOT if repo is None else Path(repo)
    if not requested.is_absolute() or "\x00" in str(requested):
        raise EvidenceError("semantic G1 repository path must be absolute")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("cannot resolve semantic G1 repository") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise EvidenceError("semantic G1 repository is not a real directory")
    return resolved


def controller_workspace(repo: Path) -> Path:
    """Bind a PASS-capable controller to the exact workflow workspace.

    ``GITHUB_WORKSPACE`` is supplied by the reviewed, exact workflow.  It is
    not inferred from ``__file__``, a caller supplied ``--repo`` value, or a
    copied checkout.  The exact textual path requirement also rejects symlink
    and ``..`` aliases before Git object lookup.
    """

    workspace_text = os.environ.get("GITHUB_WORKSPACE")
    if not workspace_text:
        raise EvidenceError("semantic G1 PASS authority requires GITHUB_WORKSPACE")
    workspace = Path(workspace_text)
    if not workspace.is_absolute() or "\x00" in workspace_text:
        raise EvidenceError("GITHUB_WORKSPACE is not an absolute closed path")
    try:
        resolved = workspace.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("cannot resolve GITHUB_WORKSPACE") from exc
    if str(workspace) != str(resolved) or workspace.is_symlink() or not resolved.is_dir():
        raise EvidenceError("GITHUB_WORKSPACE is not an exact non-symlink controller path")
    if canonical_repository(repo) != resolved:
        raise EvidenceError("semantic G1 controller repository differs from GITHUB_WORKSPACE")
    return resolved


def canonical_build_tools() -> dict[str, Path]:
    """Resolve only fixed absolute build executables, never caller PATH."""

    tools = {
        "cargo": CANONICAL_CARGO,
        "cmake": CANONICAL_CMAKE,
        "cxx": CANONICAL_CXX,
        "python": CANONICAL_PYTHON,
    }
    result: dict[str, Path] = {}
    for name, path in tools.items():
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise EvidenceError(f"canonical {name} executable is unavailable") from exc
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise EvidenceError(f"canonical {name} executable is not executable")
        result[name] = resolved
    return result


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _git_command(repo: Path, args: Sequence[str]) -> list[str]:
    if not repo.is_absolute() or repo.is_symlink():
        raise EvidenceError("Git authority repository path is not an exact absolute directory")
    return ["/usr/bin/git", "--no-replace-objects", "-C", str(repo), *args]


def _git_output(repo: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(_git_command(repo, args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0, env=_git_environment())
        return result.stdout.decode("ascii").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise EvidenceError("cannot read immutable repository identity") from exc


def _git_output_bytes(repo: Path, args: Sequence[str]) -> bytes:
    try:
        return subprocess.run(_git_command(repo, args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0, env=_git_environment()).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("cannot read reviewed repository bytes") from exc


def git_object_format(repo: Path) -> str:
    value = _git_output(canonical_repository(repo), ("rev-parse", "--show-object-format=storage"))
    if value not in GIT_OBJECT_FORMATS:
        raise EvidenceError(f"Git object format is unsupported: {value}")
    return value


def git_blob_oid(repo: Path, revision: str, relative: str) -> tuple[str, int]:
    """Return a real repository blob OID and its format width."""

    repository = canonical_repository(repo)
    object_format = git_object_format(repository)
    blob = _git_output(repository, ("rev-parse", "--verify", f"{revision}:{relative}"))
    width = GIT_OBJECT_FORMATS[object_format]
    if len(blob) != width or GIT_OID_RE.fullmatch(blob) is None:
        raise EvidenceError("Git blob OID does not match the repository object format")
    return blob, width


def verify_repository_identity(repo: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    repo = controller_workspace(repo)
    identity = _identity(expected, "repository")
    object_format = git_object_format(repo)
    oid_width = GIT_OBJECT_FORMATS[object_format]
    if len(identity["reviewed_sha"]) != oid_width:
        raise EvidenceError("reviewed Git commit width does not match the repository object format")
    replace_refs = _git_output(repo, ("for-each-ref", "--format=%(refname)", "refs/replace"))
    if replace_refs:
        raise EvidenceError("Git replacement refs are forbidden for semantic G1 authority")
    local_config = _git_output(repo, ("config", "--local", "--null", "--list"))
    safe_config = {
        "core.repositoryformatversion", "core.filemode", "core.bare", "core.logallrefupdates",
        "core.symlinks", "core.ignorecase", "core.precomposeunicode", "extensions.objectformat",
    }
    config_keys = [item.split(b"\n", 1)[0].decode("ascii") for item in local_config.encode("utf-8").split(b"\0") if item]
    if any(key not in safe_config for key in config_keys):
        raise EvidenceError("Git local configuration contains an authority-affecting or dangerous key")
    for name in ("GITHUB_SHA", "REVIEWED_SHA", "TESTED_SHA", "WORKFLOW_SHA"):
        if os.environ.get(name) != identity["reviewed_sha"]:
            raise EvidenceError(f"semantic G1 requires immutable workflow {name} to equal the reviewed candidate")
    for name in ("GITHUB_WORKFLOW_SHA", "SLLM_REVIEWED_SHA", "SLLM_TESTED_SHA"):
        value = os.environ.get(name)
        if value is not None and value != identity["reviewed_sha"]:
            raise EvidenceError(f"{name} disagrees with the reviewed semantic G1 candidate")
    if _git_output(repo, ("status", "--porcelain=v1", "--untracked-files=all")):
        raise EvidenceError("semantic G1 requires a clean immutable candidate worktree")
    commit = _git_output(repo, ("rev-parse", "--verify", "HEAD^{commit}"))
    tree = _git_output(repo, ("rev-parse", "--verify", f"{commit}^{{tree}}"))
    if len(commit) != oid_width or len(tree) != oid_width or commit != identity["reviewed_sha"]:
        raise EvidenceError("checked-out repository does not match the exact candidate")
    supplied_tree = expected.get("git_tree_oid")
    if supplied_tree is not None and str(supplied_tree) != tree:
        raise EvidenceError("caller-supplied Git tree OID differs from the internally recomputed tree")
    return {**identity, "git_tree_oid": tree, "git_object_format": object_format, "git_oid_width": oid_width, "worktree_clean": True, "revision_input": "full-sha"}


def actual_repository_identity(repo: Path = ROOT) -> dict[str, Any]:
    repo = controller_workspace(repo)
    commit = _git_output(repo, ("rev-parse", "--verify", "HEAD^{commit}"))
    return verify_repository_identity(repo, {"reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit})


def approved_repository_file(repo: Path, identity: Mapping[str, Any], relative: str, label: str) -> SealedDescriptor:
    """Seal the reviewed Git object itself, never reopen its worktree path.

    A clean checkout proves the selected candidate at the controller gate, but
    it is not an immutable input source after that point.  The worker and
    controller therefore consume a controller-held copy of the named Git
    object's bytes; replacing a path in the checkout cannot replace code that
    will execute or be audited.
    """

    repo = controller_workspace(repo)
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise EvidenceError("approved repository file path is unsafe")
    candidate = _identity(identity, "approved source")
    expected_bytes = _git_output_bytes(repo, ("show", f"{candidate['reviewed_sha']}:{relative}"))
    path = repo / relative
    try:
        path.relative_to(repo)
    except ValueError as exc:
        raise EvidenceError("approved repository file escapes the candidate") from exc
    return snapshot_bytes(expected_bytes, logical_path=str(path), label=label)


def _static_executable_record(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate a reviewed host executable record without caller PATH use."""

    record = _validate_digest_record(expected, label)
    if Path(record["path"]) != path:
        raise EvidenceError(f"{label} path is not the reviewed absolute executable")
    observed = file_identity(path, label)
    if observed != record:
        raise EvidenceError(f"{label} bytes or resolved executable path drifted")
    if not os.access(path, os.X_OK):
        raise EvidenceError(f"{label} is not executable")
    return observed


def reviewed_authority(repo: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Create the independently auditable immutable G1 authority object.

    Every project byte that can influence a semantic-G1 PASS is read from the
    reviewed Git object, checked against the current checkout once, and named
    by its Git blob ID.  The emitted object therefore has no ``__file__`` or
    copied-checkout self-authentication field for an auditor to trust.
    """

    repo = controller_workspace(repo)
    candidate = verify_repository_identity(repo, identity)
    source_records: list[dict[str, Any]] = []
    for relative in AUTHORITY_SOURCE_FILES:
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise EvidenceError("semantic G1 reviewed source path is unsafe")
        source_bytes = _git_output_bytes(repo, ("show", f"{candidate['reviewed_sha']}:{relative}"))
        blob_oid, blob_width = git_blob_oid(repo, candidate["reviewed_sha"], relative)
        if blob_width != candidate["git_oid_width"] or not source_bytes:
            raise EvidenceError("reviewed semantic G1 source object is malformed")
        source_records.append({
            "path": relative,
            "git_blob_oid": blob_oid,
            "size_bytes": len(source_bytes),
            "sha256": sha256_bytes(source_bytes),
        })
    by_path = {str(record["path"]): record for record in source_records}
    if len(by_path) != len(AUTHORITY_SOURCE_FILES) or set(by_path) != set(AUTHORITY_SOURCE_FILES):
        raise EvidenceError("semantic G1 reviewed authority source set is not closed")
    controller = by_path["ci/tools/orchestrate_rmsnorm_g1_evidence.py"]
    workflow = by_path[WORKFLOW_PATH]
    return {
        "authority_version": AUTHORITY_VERSION,
        "candidate": candidate,
        "controller": controller,
        "workflow": workflow,
        "sources": [by_path[path] for path in AUTHORITY_SOURCE_FILES],
        "executables": {
            "python": _static_executable_record(CANONICAL_PYTHON, CONTROLLER_PYTHON_RECORD, "controller Python"),
            "compiler": _static_executable_record(Path(COMPILER_LOGICAL_PATH), COMPILER_SOURCE_RECORD, "ROCm compiler"),
            "client_interpreter": _static_executable_record(CANONICAL_PYTHON, COMPILER_CLIENT_INTERPRETER_RECORD, "compiler broker client interpreter"),
        },
        "toolchain": {
            name: file_identity(path, f"semantic G1 toolchain {name}")
            for name, path in CANONICAL_TOOL_PATHS.items()
        },
    }


def authority_contract_hashes(authority: Mapping[str, Any]) -> dict[str, str]:
    """Derive all schema/matrix hashes from the reviewed authority object."""

    sources = authority.get("sources")
    if not isinstance(sources, list):
        raise EvidenceError("semantic G1 authority has no reviewed sources")
    records = {entry.get("path"): entry for entry in sources if isinstance(entry, Mapping)}
    if len(records) != len(sources):
        raise EvidenceError("semantic G1 authority source list is malformed")
    values = {
        "matrix_manifest_sha256": MATRIX_MANIFEST,
        "matrix_schema_sha256": MATRIX_SCHEMA,
        "artifact_schema_sha256": ARTIFACT_SCHEMA,
        "report_schema_sha256": REPORT_SCHEMA,
        "aggregate_schema_sha256": AGGREGATE_SCHEMA,
        "model_lock_sha256": MODEL_LOCK,
    }
    result: dict[str, str] = {}
    for name, path in values.items():
        record = records.get(path)
        if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str) or SHA256_RE.fullmatch(str(record["sha256"])) is None:
            raise EvidenceError("semantic G1 authority is missing a reviewed contract byte hash")
        result[name] = str(record["sha256"])
    return result


def approved_python_interpreter() -> SealedDescriptor:
    """Use only the fixed system interpreter path chosen by the controller."""

    _static_executable_record(CANONICAL_PYTHON, CONTROLLER_PYTHON_RECORD, "controller Python")
    return snapshot_file(CANONICAL_PYTHON, CONTROLLER_PYTHON_RECORD, "controller Python interpreter")


def manifest_hashes(repo: Path = ROOT) -> dict[str, str]:
    repo = canonical_repository(repo)
    def digest(relative: str) -> str:
        data = _reviewed_contract_bytes(repo, relative)
        return sha256_bytes(data) if data is not None else sha256_file(repo / relative)
    return {
        "matrix_manifest_sha256": digest(MATRIX_MANIFEST),
        "matrix_schema_sha256": digest(MATRIX_SCHEMA),
        "artifact_schema_sha256": digest(ARTIFACT_SCHEMA),
        "report_schema_sha256": digest(REPORT_SCHEMA),
        "aggregate_schema_sha256": digest(AGGREGATE_SCHEMA),
        "model_lock_sha256": digest(MODEL_LOCK),
    }


def validate_matrix(repo: Path = ROOT) -> dict[str, Any]:
    repo = canonical_repository(repo)
    matrix_bytes = _reviewed_contract_bytes(repo, MATRIX_MANIFEST)
    matrix = read_json_bytes(matrix_bytes, MATRIX_MANIFEST) if matrix_bytes is not None else read_json(repo / MATRIX_MANIFEST)
    validate_schema(matrix, _schema(repo, MATRIX_SCHEMA), "semantic G1 matrix")
    if matrix.get("suite_id") != MATRIX_SUITE_ID or matrix.get("scope") != EXPECTED_SCOPE or matrix.get("command") != EXPECTED_COMMAND:
        raise EvidenceError("semantic G1 matrix scope/suite/command drifted")
    oracle = matrix.get("oracle")
    model_lock_bytes = _reviewed_contract_bytes(repo, MODEL_LOCK)
    model_lock_digest = sha256_bytes(model_lock_bytes) if model_lock_bytes is not None else sha256_file(repo / MODEL_LOCK)
    if not isinstance(oracle, Mapping) or oracle.get("model_hidden_size") != MODEL_HIDDEN_SIZE or oracle.get("epsilon") != MODEL_EPSILON_TEXT or oracle.get("model_lock_path") != MODEL_LOCK or oracle.get("model_lock_sha256") != model_lock_digest or model_lock_digest != MODEL_LOCK_SHA256:
        raise EvidenceError("semantic G1 matrix epsilon/hidden-size/model-lock authority drifted")
    model_lock = read_json_bytes(model_lock_bytes, MODEL_LOCK) if model_lock_bytes is not None else read_json(repo / MODEL_LOCK)
    normalization = model_lock.get("model", {}).get("slice_contract", {}).get("normalization", {}) if isinstance(model_lock, Mapping) else {}
    text_config = model_lock.get("model", {}).get("architecture", {}).get("text_config", {}) if isinstance(model_lock, Mapping) else {}
    if normalization.get("epsilon") != MODEL_EPSILON_TEXT or text_config.get("hidden_size") != MODEL_HIDDEN_SIZE:
        raise EvidenceError("semantic G1 model-lock does not independently bind epsilon and hidden size")
    if matrix.get("case_manifest", {}).get("cases") != EXPECTED_CASES:
        raise EvidenceError("semantic G1 case set drifted")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or [row.get("row_id") for row in rows if isinstance(row, dict)] != list(ROWS):
        raise EvidenceError("semantic G1 matrix must retain the canonical ordered two rows")
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceError("semantic G1 matrix row is not an object")
        target = row.get("target")
        if target not in TARGETS:
            raise EvidenceError("semantic G1 matrix target is not canonical")
        expected = {"row_id": f"rmsnorm-semantic-g1-{target}", "target": target, **EXPECTED_BINDINGS[target]}
        if any(row.get(key) != value for key, value in expected.items()) or row.get("scope") != EXPECTED_SCOPE or row.get("codegen") != EXPECTED_CODEGEN[target]:
            raise EvidenceError("semantic G1 matrix canonical device binding drifted")
    return matrix


def row_by_id(matrix: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    for row in matrix.get("rows", []):
        if isinstance(row, Mapping) and row.get("row_id") == row_id:
            return dict(row)
    raise EvidenceError("unknown semantic G1 matrix row")


def _validate_authority_document(authority: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a report artifact to the controller's reviewed authority object."""

    if not isinstance(authority, Mapping):
        raise EvidenceError("semantic G1 document authority is malformed")
    required = {"authority_version", "candidate", "controller", "workflow", "sources", "executables", "toolchain"}
    if set(authority) != required or authority.get("authority_version") != AUTHORITY_VERSION:
        raise EvidenceError("semantic G1 document authority is not the closed reviewed authority object")
    _validate_candidate_document(authority.get("candidate", {}), identity, "semantic G1 authority")
    sources = authority.get("sources")
    if not isinstance(sources, list) or len(sources) != len(AUTHORITY_SOURCE_FILES):
        raise EvidenceError("semantic G1 authority source list is incomplete")
    expected_paths = list(AUTHORITY_SOURCE_FILES)
    observed_paths: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"path", "git_blob_oid", "size_bytes", "sha256"}:
            raise EvidenceError("semantic G1 authority source record is malformed")
        if not isinstance(source.get("path"), str) or not isinstance(source.get("git_blob_oid"), str):
            raise EvidenceError("semantic G1 authority source path/blob is malformed")
        if GIT_OID_RE.fullmatch(str(source["git_blob_oid"])) is None or SHA256_RE.fullmatch(str(source.get("sha256", ""))) is None:
            raise EvidenceError("semantic G1 authority source digest is malformed")
        if isinstance(source.get("size_bytes"), bool) or not isinstance(source.get("size_bytes"), int) or int(source["size_bytes"]) < 1:
            raise EvidenceError("semantic G1 authority source size is malformed")
        observed_paths.append(str(source["path"]))
    if observed_paths != expected_paths:
        raise EvidenceError("semantic G1 authority sources are not the exact reviewed ordered set")
    by_path = {str(source["path"]): source for source in sources}
    if authority.get("controller") != by_path.get("ci/tools/orchestrate_rmsnorm_g1_evidence.py") or authority.get("workflow") != by_path.get(WORKFLOW_PATH):
        raise EvidenceError("semantic G1 authority controller/workflow anchors are not reviewed source records")
    executables = authority.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"python", "compiler", "client_interpreter"}:
        raise EvidenceError("semantic G1 authority executable records are malformed")
    if executables.get("python") != CONTROLLER_PYTHON_RECORD or executables.get("compiler") != COMPILER_SOURCE_RECORD or executables.get("client_interpreter") != COMPILER_CLIENT_INTERPRETER_RECORD:
        raise EvidenceError("semantic G1 authority executable records do not equal reviewed fixed bytes")
    toolchain = authority.get("toolchain")
    if not isinstance(toolchain, Mapping) or set(toolchain) != set(CANONICAL_TOOL_KEYS):
        raise EvidenceError("semantic G1 authority toolchain record is not closed")
    for name, path in CANONICAL_TOOL_PATHS.items():
        record = _validate_digest_record(toolchain[name], f"semantic G1 toolchain {name}")
        if Path(record["path"]) != path or Path(record["resolved_path"]) != path.resolve(strict=False):
            raise EvidenceError(f"semantic G1 toolchain executable {name} path drifted")
    return dict(authority)


def validate_report_document(
    report: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo: Path = ROOT,
    authority: Mapping[str, Any] | None = None,
    artifact_facts: Mapping[str, Any] | None = None,
) -> None:
    """Close every report field against the canonical matrix and manifests."""

    repo = canonical_repository(repo)
    canonical_matrix = validate_matrix(repo)
    canonical_row = row_by_id(canonical_matrix, str(row.get("row_id", "")))
    if dict(row) != canonical_row:
        raise EvidenceError("semantic G1 report was not constructed from a canonical matrix row")
    validate_schema(dict(report), _schema(repo, REPORT_SCHEMA), "controller-derived semantic G1 report")
    if report.get("row_id") != canonical_row["row_id"] or report.get("target") != canonical_row["target"]:
        raise EvidenceError("semantic G1 report row/target binding drifted")
    expected_device = {key: canonical_row[key] for key in ("bdf", "uuid", "target", "physical_hip_index", "logical_device_index")}
    if report.get("device") != expected_device:
        raise EvidenceError("semantic G1 report device does not equal its canonical matrix row")
    for name in ("health_pre", "health_post", "process_pre", "process_post"):
        fact = report.get(name)
        if not isinstance(fact, Mapping) or fact.get("device") != expected_device:
            raise EvidenceError(f"semantic G1 report {name} is not bound to its row device")
    _validate_candidate_document(report.get("candidate", {}), identity, "semantic G1 report")
    if authority is None or report.get("authority") != _require_live_authority(authority, identity, repo, "semantic G1 report"):
        raise EvidenceError("semantic G1 report authority does not equal the live reviewed controller authority")
    if report.get("contracts") != authority_contract_hashes(authority) or report.get("scope") != EXPECTED_SCOPE:
        raise EvidenceError("semantic G1 report contracts/scope drifted")
    if report.get("resource_counts") != EXPECTED_ROW_RESOURCE_COUNTS:
        raise EvidenceError("semantic G1 report resource counts are not the fixed semantic protocol counts")
    artifact = report.get("artifact")
    compiler = report.get("compiler_execution")
    if not isinstance(artifact, Mapping) or not isinstance(compiler, Mapping):
        raise EvidenceError("semantic G1 report artifact/compiler binding is malformed")
    if artifact_facts is not None and dict(artifact) != dict(artifact_facts):
        raise EvidenceError("semantic G1 report artifact facts do not equal the controller-captured sealed bundle")
    _validate_serialized_compiler_execution(compiler)
    if artifact.get("compiler_execution_sha256") != sha256_json(compiler):
        raise EvidenceError("semantic G1 report compiler transcript digest is not bound to its artifact")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_CASES):
        raise EvidenceError("semantic G1 report case collection is incomplete")
    for order, (expected_case, observed_case) in enumerate(zip(EXPECTED_CASES, cases, strict=True)):
        if not isinstance(observed_case, Mapping) or {key: observed_case.get(key) for key in ("order", "id", "rows", "n", "classification", "nonfinite_input")} != {"order": order, **expected_case}:
            raise EvidenceError("semantic G1 report case dimensions/order drifted")
        if "response_b64" in observed_case:
            raise EvidenceError("semantic G1 report must retain only the raw-frame digest, never raw frame bytes")
        if observed_case.get("resource_counts") != EXPECTED_CASE_RESOURCE_COUNTS or observed_case.get("dispatch_count") != 1:
            raise EvidenceError("semantic G1 report case resource/dispatch count drifted")
        evidence = observed_case.get("response_evidence")
        if not isinstance(evidence, Mapping):
            raise EvidenceError("semantic G1 report case has no retained response evidence binding")
        expected_path = f"rows/{canonical_row['row_id']}/raw/case-{order}.bin"
        expected_sidecar = expected_path + ".sha256"
        if (
            evidence.get("path") != expected_path
            or evidence.get("sidecar_path") != expected_sidecar
            or evidence.get("candidate_sha256") != sha256_json(report["candidate"])
            or evidence.get("row_id") != canonical_row["row_id"]
            or evidence.get("case_id") != expected_case["id"]
            or evidence.get("order") != order
            or evidence.get("sha256") != observed_case.get("response_sha256")
            or not isinstance(evidence.get("size_bytes"), int)
            or not 1 <= int(evidence["size_bytes"]) <= MAX_OUTPUT
            or not isinstance(evidence.get("sidecar_sha256"), str)
            or SHA256_RE.fullmatch(evidence["sidecar_sha256"]) is None
        ):
            raise EvidenceError("semantic G1 response evidence is not bound to candidate/row/case identity")


def validate_aggregate_document(document: Mapping[str, Any], *, identity: Mapping[str, Any], repo: Path = ROOT, authority: Mapping[str, Any] | None = None) -> None:
    """Validation-only aggregate contract; it never emits or returns PASS."""

    repo = canonical_repository(repo)
    validate_matrix(repo)
    validate_schema(dict(document), _schema(repo, AGGREGATE_SCHEMA), "semantic G1 aggregate")
    _validate_candidate_document(document.get("candidate", {}), identity, "semantic G1 aggregate")
    if authority is None or document.get("authority") != _require_live_authority(authority, identity, repo, "semantic G1 aggregate"):
        raise EvidenceError("semantic G1 aggregate authority does not equal the live reviewed controller authority")
    if document.get("contracts") != authority_contract_hashes(authority) or document.get("scope") != EXPECTED_SCOPE:
        raise EvidenceError("semantic G1 aggregate contracts/scope drifted")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ROWS):
        raise EvidenceError("semantic G1 aggregate rows are incomplete")
    for row_id, record in zip(ROWS, rows, strict=True):
        if not isinstance(record, Mapping):
            raise EvidenceError("semantic G1 aggregate row is malformed")
        target = row_id.rsplit("-", 1)[1]
        if record.get("row_id") != row_id or record.get("target") != target or record.get("resource_counts") != EXPECTED_ROW_RESOURCE_COUNTS:
            raise EvidenceError("semantic G1 aggregate row-to-target/resource binding drifted")
        evidence = record.get("response_evidence")
        if not isinstance(evidence, list) or len(evidence) != len(EXPECTED_CASES):
            raise EvidenceError("semantic G1 aggregate response evidence collection is incomplete")
        candidate_digest = sha256_json(document["candidate"])
        for order, (expected_case, item) in enumerate(zip(EXPECTED_CASES, evidence, strict=True)):
            expected_path = f"rows/{row_id}/raw/case-{order}.bin"
            if not isinstance(item, Mapping) or item.get("path") != expected_path or item.get("sidecar_path") != expected_path + ".sha256" or item.get("candidate_sha256") != candidate_digest or item.get("row_id") != row_id or item.get("case_id") != expected_case["id"] or item.get("order") != order:
                raise EvidenceError("semantic G1 aggregate response evidence is misbound")
        compiler = record.get("compiler_execution")
        if not isinstance(compiler, Mapping):
            raise EvidenceError("semantic G1 aggregate row has no auditable compiler transcript")
        _validate_serialized_compiler_execution(compiler)
        if record.get("compiler_execution_sha256") != sha256_json(compiler):
            raise EvidenceError("semantic G1 aggregate compiler transcript digest drifted")


def recompute_saved_response_evidence(
    evidence_root: Path,
    report: Mapping[str, Any],
    *,
    aggregate: Mapping[str, Any] | None = None,
    expected_candidate: Mapping[str, Any] | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Recompute a saved synthetic row from only uploaded response preimages.

    This is intentionally validation-only: it returns recomputed facts and
    never creates or promotes a PASS aggregate.  The response and its
    sidecar are the only runtime bytes needed; the request preimage is
    deterministically regenerated from the canonical row/case contract.
    """

    from run_rmsnorm_g1_runtime import independent_rmsnorm_oracle, parse_response, _bf16_to_f32, _f32_to_bf16, encode_request  # noqa: PLC0415

    root = canonical_repository(repo)
    matrix = validate_matrix(root)
    if not isinstance(evidence_root, Path) or not evidence_root.is_absolute() or evidence_root.is_symlink() or not evidence_root.is_dir():
        raise EvidenceError("offline semantic G1 evidence root is not a private regular directory")
    row_id = report.get("row_id")
    if row_id not in ROWS or report.get("target") != str(row_id).rsplit("-", 1)[1] or report.get("state") != "PASS":
        raise EvidenceError("offline semantic G1 report row/state binding is invalid")
    row = row_by_id(matrix, str(row_id))
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        raise EvidenceError("offline semantic G1 report has no candidate identity")
    if expected_candidate is not None and dict(candidate) != dict(expected_candidate):
        raise EvidenceError("offline semantic G1 report candidate identity is not the expected candidate")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_CASES):
        raise EvidenceError("offline semantic G1 report case collection is incomplete")
    if aggregate is not None:
        aggregate_rows = aggregate.get("rows")
        aggregate_row = next((item for item in aggregate_rows if isinstance(item, Mapping) and item.get("row_id") == row_id), None) if isinstance(aggregate_rows, list) else None
        if not isinstance(aggregate_row, Mapping) or aggregate_row.get("response_evidence") != [case.get("response_evidence") for case in cases]:
            raise EvidenceError("offline semantic G1 aggregate does not bind the report response evidence")

    def input_bytes(case: Mapping[str, Any]) -> tuple[bytes, bytes, float]:
        rows_count, width = int(case["rows"]), int(case["n"])
        classification = str(case["classification"])
        source = str(case["nonfinite_input"])
        special = {"nan": float("nan"), "posinf": float("inf"), "neginf": -float("inf")}.get(classification)
        if special is not None and source not in {"activation", "raw_scale"} or special is None and source != "none":
            raise EvidenceError("offline semantic G1 case does not name a valid nonfinite input source")
        activation = bytearray()
        for index in range(rows_count * width):
            value = special if index == 0 and source == "activation" else ((index * 37 + int(row["seed"])) % 257 - 128) / 32.0
            activation.extend(struct.pack("<H", _f32_to_bf16(value)))
        raw_scale = bytearray()
        for index in range(width):
            value = special if index == 0 and source == "raw_scale" else ((index * 19 + int(row["seed"])) % 65 - 32) / 128.0
            raw_scale.extend(struct.pack("<H", _f32_to_bf16(value)))
        epsilon = MODEL_EPSILON
        return bytes(activation), bytes(raw_scale), epsilon

    def numerics(actual: bytes, expected: bytes) -> dict[str, Any]:
        max_abs, max_rel, nan_count, inf_count = 0.0, 0.0, 0, 0
        for actual_bits, expected_bits in zip(struct.iter_unpack("<H", actual), struct.iter_unpack("<H", expected), strict=True):
            observed, reference = _bf16_to_f32(actual_bits[0]), _bf16_to_f32(expected_bits[0])
            if math.isnan(reference) or math.isnan(observed):
                if not (math.isnan(reference) and math.isnan(observed)):
                    raise EvidenceError("offline semantic G1 NaN classification differs")
                nan_count += 1
            elif math.isinf(reference) or math.isinf(observed):
                if not (math.isinf(reference) and math.isinf(observed) and math.copysign(1.0, reference) == math.copysign(1.0, observed)):
                    raise EvidenceError("offline semantic G1 Inf classification differs")
                inf_count += 1
            else:
                absolute = abs(observed - reference)
                relative = absolute / abs(reference) if reference else absolute
                max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
                if absolute > 0.0078125 + 0.015625 * abs(reference):
                    raise EvidenceError("offline semantic G1 output exceeds the registered tolerance")
        return {"tolerance_id": "rmsnorm-bf16-f32-output-v1", "atol": 0.0078125, "rtol": 0.015625, "max_abs_error": max_abs, "max_rel_error": max_rel, "nan_count": nan_count, "inf_count": inf_count}

    raw_parts: list[bytes] = []
    totals = {name: 0 for name in EXPECTED_CASE_RESOURCE_COUNTS}
    evidence_paths: list[str] = []
    for order, (expected_case, observed_case) in enumerate(zip(EXPECTED_CASES, cases, strict=True)):
        if not isinstance(observed_case, Mapping) or {key: observed_case.get(key) for key in ("order", "id", "rows", "n", "classification", "nonfinite_input")} != {"order": order, **expected_case}:
            raise EvidenceError("offline semantic G1 case identity/order drifted")
        evidence = observed_case.get("response_evidence")
        expected_path = f"rows/{row_id}/raw/case-{order}.bin"
        if not isinstance(evidence, Mapping) or evidence.get("path") != expected_path or evidence.get("sidecar_path") != expected_path + ".sha256" or evidence.get("candidate_sha256") != sha256_json(candidate) or evidence.get("row_id") != row_id or evidence.get("case_id") != expected_case["id"] or evidence.get("order") != order:
            raise EvidenceError("offline semantic G1 response evidence is misbound")
        response_path = evidence_root / expected_path
        sidecar_path = evidence_root / (expected_path + ".sha256")
        if response_path.is_symlink() or sidecar_path.is_symlink() or not response_path.is_file() or not sidecar_path.is_file():
            raise EvidenceError("offline semantic G1 response or sidecar is missing or not regular")
        response = response_path.read_bytes()
        sidecar = sidecar_path.read_bytes()
        response_digest = sha256_bytes(response)
        if not 1 <= len(response) <= MAX_OUTPUT or evidence.get("size_bytes") != len(response) or evidence.get("sha256") != response_digest or observed_case.get("response_sha256") != response_digest or evidence.get("sidecar_sha256") != sha256_bytes(sidecar) or sidecar != _sidecar_text(response_digest, response_path.name):
            raise EvidenceError("offline semantic G1 response/sidecar digest or size is invalid")
        activation, scale, epsilon = input_bytes(expected_case)
        request = encode_request((int(expected_case["rows"]), int(expected_case["n"])), epsilon, activation, scale)
        parsed = parse_response(response, expected_target=str(row["target"]), expected_device_index=int(row["logical_device_index"]), expected_shape=(int(expected_case["rows"]), int(expected_case["n"])), expected_epsilon=epsilon)
        oracle = independent_rmsnorm_oracle(activation, scale, int(expected_case["rows"]), int(expected_case["n"]), epsilon)
        if observed_case.get("request_sha256") != sha256_bytes(request) or parsed["resource_counts"] != EXPECTED_CASE_RESOURCE_COUNTS or observed_case.get("resource_counts") != parsed["resource_counts"] or observed_case.get("dispatch_count") != parsed["dispatch_count"] or observed_case.get("dispatch_id") != parsed["dispatch_id"] or observed_case.get("kernel_symbol") != parsed["kernel_symbol"] or observed_case.get("device_symbol") != parsed["device_symbol"] or observed_case.get("numerics") != numerics(parsed["output"], oracle):
            raise EvidenceError("offline semantic G1 recomputation differs from the retained report")
        for name, value in parsed["resource_counts"].items():
            totals[name] += int(value)
        raw_parts.append(response)
        evidence_paths.append(expected_path)
    expected_total = {name: value * len(EXPECTED_CASES) for name, value in EXPECTED_CASE_RESOURCE_COUNTS.items()}
    if totals != expected_total or report.get("resource_counts") != totals or report.get("raw_frame_sha256") != sha256_bytes(b"".join(raw_parts)):
        raise EvidenceError("offline semantic G1 row totals are not recomputable from retained responses")
    return {"row_id": row_id, "target": row["target"], "case_count": len(cases), "raw_frame_sha256": sha256_bytes(b"".join(raw_parts)), "resource_counts": totals, "evidence_paths": evidence_paths}


def _validate_candidate_document(candidate: Mapping[str, Any], identity: Mapping[str, Any], label: str) -> None:
    if dict(candidate) != _candidate_document(identity, label):
        raise EvidenceError(f"{label} candidate is not the exact immutable controller candidate")


def _require_live_authority(
    authority: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo: Path,
    label: str,
) -> dict[str, Any]:
    """Recompute authority from the exact controller workspace before use."""

    if os.environ.get("GITHUB_WORKSPACE") != str(repo):
        raise EvidenceError(f"{label} has no live controller workspace authority")
    try:
        expected = reviewed_authority(repo, identity)
    except (EvidenceError, OSError) as exc:
        raise EvidenceError(f"{label} could not recompute live reviewed authority") from exc
    if dict(authority) != expected:
        raise EvidenceError(f"{label} authority is fabricated or stale")
    return _validate_authority_document(authority, identity)


def _validate_serialized_compiler_execution(compiler: Mapping[str, Any]) -> dict[str, Any]:
    """Validate parent issuance and one-shot consumption of whole actions."""

    required = {"protocol", "event_protocol", "source", "client", "exec_helper", "session", "request_count", "events_sha256", "expected_recipe_keys", "closure", "actions", "events"}
    if not isinstance(compiler, Mapping) or set(compiler) != required:
        raise EvidenceError("compiler execution is not a closed exact-action transcript")
    if compiler.get("protocol") != "parent-owned-exact-action-broker-v1" or compiler.get("event_protocol") != EXACT_ACTION_PROTOCOL:
        raise EvidenceError("compiler execution does not use the exact-action protocol")
    for name in ("source", "client", "exec_helper"):
        value = compiler.get(name)
        if not isinstance(value, Mapping):
            raise EvidenceError(f"exact-action {name} record is malformed")
        _validate_digest_record(value, f"exact-action {name}")
    if dict(compiler["source"]) != COMPILER_SOURCE_RECORD:
        raise EvidenceError("exact-action compiler is not the reviewed sealed ROCm compiler")
    expected_recipe_keys = compiler.get("expected_recipe_keys")
    if (
        not isinstance(expected_recipe_keys, list)
        or not expected_recipe_keys
        or len(expected_recipe_keys) != len(set(expected_recipe_keys))
        or any(not isinstance(value, str) or not value or "\0" in value for value in expected_recipe_keys)
    ):
        raise EvidenceError("semantic G1 exact-action transcript has no closed reviewed recipe set")
    actions = compiler.get("actions")
    if not isinstance(actions, list) or not actions:
        raise EvidenceError("exact-action transcript lacks issued actions")
    issued: dict[str, dict[str, Any]] = {}
    issued_recipe_keys: set[str] = set()
    for item in actions:
        if not isinstance(item, Mapping) or set(item) != {"recipe_key", "action_id", "action_digest", "state", "issued_at_ns", "consumed_at_ns", "manifest"}:
            raise EvidenceError("exact-action issuance record is not closed")
        manifest = _validate_exact_action(item["manifest"])
        if (
            not isinstance(item.get("recipe_key"), str)
            or item["recipe_key"] not in expected_recipe_keys
            or item["recipe_key"] in issued_recipe_keys
            or item["action_id"] != manifest["action_id"]
            or item["action_digest"] != manifest["manifest_digest"]
            or item["action_id"] in issued
        ):
            raise EvidenceError("exact-action issuance identity/digest is invalid")
        executable = manifest["executable"]
        if (
            any(executable.get(name) != compiler["source"].get(name) for name in ("path", "resolved_path", "size_bytes", "sha256"))
            or isinstance(executable.get("device"), bool)
            or not isinstance(executable.get("device"), int)
            or isinstance(executable.get("inode"), bool)
            or not isinstance(executable.get("inode"), int)
            or executable.get("seals") != REQUIRED_SEALS
            or not manifest["argv0"].startswith("/proc/self/fd/")
        ):
            raise EvidenceError("exact-action issuance is not bound to the sealed compiler/argv0 identity")
        if item["state"] not in {"issued", "consumed"} or not isinstance(item["issued_at_ns"], int) or item["issued_at_ns"] < 1 or (item["consumed_at_ns"] is not None and (not isinstance(item["consumed_at_ns"], int) or item["consumed_at_ns"] < item["issued_at_ns"])):
            raise EvidenceError("exact-action issuance state/timestamps are invalid")
        if (item["state"] == "consumed") != (item["consumed_at_ns"] is not None):
            raise EvidenceError("exact-action issuance/consumption state is inconsistent")
        issued[item["action_id"]] = dict(item)
        issued_recipe_keys.add(item["recipe_key"])
    if issued_recipe_keys != set(expected_recipe_keys) or len(actions) != len(expected_recipe_keys):
        raise EvidenceError("semantic G1 exact-action transcript omitted or duplicated a reviewed recipe")
    events = compiler.get("events")
    if not isinstance(events, list) or len(events) != compiler.get("request_count") or len(events) > 4096 or len(canonical_bytes(compiler)) > COMPILER_TRANSCRIPT_MAX_BYTES:
        raise EvidenceError("exact-action event transcript is malformed")
    consumed: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        required_event = {"sequence", "request_nonce", "observation_nonce", "client_observation", "client_binding", "action_id", "action_digest", "action_manifest", "request_frame_sha256", "response_frame_sha256", "ack_frame_sha256", "compiler_source_sha256", "compiler", "started_at_ns", "finished_at_ns", "consumed", "acknowledged"}
        if not isinstance(event, Mapping) or set(event) != required_event or event.get("sequence") != index or event.get("consumed") is not True or event.get("acknowledged") is not True:
            raise EvidenceError("exact-action event is missing, reordered, or unconsumed")
        action_id = event.get("action_id")
        manifest = _validate_exact_action(event.get("action_manifest"))
        if not isinstance(action_id, str) or action_id in consumed or action_id not in issued or event.get("action_digest") != manifest["manifest_digest"] or issued[action_id]["manifest"] != manifest or issued[action_id]["state"] != "consumed":
            raise EvidenceError("exact-action event is not bound to one issued/consumed manifest")
        consumed.add(action_id)
        if (
            event.get("compiler_source_sha256") != compiler["source"]["sha256"]
            or not isinstance(event.get("request_nonce"), str)
            or SHA256_RE.fullmatch(event["request_nonce"]) is None
            or not isinstance(event.get("observation_nonce"), str)
            or SHA256_RE.fullmatch(event["observation_nonce"]) is None
            or event["observation_nonce"] == event["request_nonce"]
        ):
            raise EvidenceError("exact-action event source/nonce is malformed")
        binding = event.get("client_binding")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"pid", "starttime", "uid", "gid"}
            or any(isinstance(binding.get(name), bool) or not isinstance(binding.get(name), int) or binding[name] < 1 for name in ("pid", "starttime"))
            or any(isinstance(binding.get(name), bool) or not isinstance(binding.get(name), int) or binding[name] < 0 for name in ("uid", "gid"))
        ):
            raise EvidenceError("exact-action client binding is malformed")
        client_observation = event.get("client_observation")
        if (
            not isinstance(client_observation, Mapping)
            or set(client_observation) != {"observation_nonce", "argv", "cwd", "environment_sha256", "client_binding"}
            or client_observation.get("observation_nonce") != event["observation_nonce"]
            or client_observation.get("client_binding") != binding
            or not isinstance(client_observation.get("argv"), list)
            or not client_observation["argv"]
            or any(not isinstance(value, str) or not value or "\0" in value for value in client_observation["argv"])
            or not isinstance(client_observation.get("cwd"), str)
            or not Path(client_observation["cwd"]).is_absolute()
            or not isinstance(client_observation.get("environment_sha256"), str)
            or SHA256_RE.fullmatch(client_observation["environment_sha256"]) is None
            or client_observation["argv"] != manifest["argv"]
            or client_observation["cwd"] != manifest["cwd"]["path"]
        ):
            raise EvidenceError("exact-action client observation is not closed and bound")
        if (
            not isinstance(event.get("started_at_ns"), int)
            or isinstance(event.get("started_at_ns"), bool)
            or event["started_at_ns"] < 1
            or not isinstance(event.get("finished_at_ns"), int)
            or isinstance(event.get("finished_at_ns"), bool)
            or event["finished_at_ns"] < event["started_at_ns"]
            or any(not isinstance(event.get(name), str) or SHA256_RE.fullmatch(event[name]) is None for name in ("request_frame_sha256", "response_frame_sha256", "ack_frame_sha256"))
        ):
            raise EvidenceError("exact-action event timestamps/frame digests are malformed")
        result = event.get("compiler")
        result_keys = {"pid", "starttime", "ppid", "pgrp", "status", "exit_code", "stdout_b64", "stderr_b64", "stdout_sha256", "stderr_sha256", "duration_ns", "timed_out", "crashed", "invocation", "kernel_limits", "action_id", "action_digest", "exec_identity"}
        if not isinstance(result, Mapping) or set(result) != result_keys or result.get("action_id") != action_id or result.get("action_digest") != manifest["manifest_digest"]:
            raise EvidenceError("exact-action compiler result is not bound to its manifest")
        if result.get("status") not in {"ok", "failed"} or result.get("status") != ("ok" if result.get("exit_code") == 0 else "failed") or not isinstance(result.get("exit_code"), int) or not isinstance(result.get("timed_out"), bool) or not isinstance(result.get("crashed"), bool):
            raise EvidenceError("exact-action compiler status is malformed")
        if any(isinstance(result.get(name), bool) or not isinstance(result.get(name), int) or result[name] < 1 for name in ("pid", "starttime", "ppid", "pgrp", "duration_ns")):
            raise EvidenceError("exact-action compiler process identity/duration is malformed")
        decoded: dict[str, bytes] = {}
        for encoded_name, digest_name in (("stdout_b64", "stdout_sha256"), ("stderr_b64", "stderr_sha256")):
            try:
                decoded[encoded_name] = base64.b64decode(result[encoded_name], validate=True)
            except (TypeError, ValueError) as exc:
                raise EvidenceError("exact-action compiler output encoding is malformed") from exc
            if len(decoded[encoded_name]) > 256 * 1024 or result.get(digest_name) != sha256_bytes(decoded[encoded_name]):
                raise EvidenceError("exact-action compiler output digest is malformed")
        invocation = result.get("invocation")
        if not isinstance(invocation, Mapping) or set(invocation) != {"action_manifest", "materialized_outputs", "sealed_input_view"} or invocation.get("action_manifest") != manifest or not isinstance(invocation.get("materialized_outputs"), list):
            raise EvidenceError("exact-action invocation does not preserve its whole manifest")
        sealed_argv = _validate_sealed_input_view(invocation["sealed_input_view"], manifest)
        if [record.get("path") for record in invocation["materialized_outputs"]] != [record["path"] for record in manifest["outputs"]]:
            raise EvidenceError("exact-action materialized outputs differ from the issued output set")
        for record in invocation["materialized_outputs"]:
            _validate_digest_record(record, "exact-action materialized output")
        limits = result.get("kernel_limits")
        if limits != {
            "address_space_bytes": 8 * 1024 * 1024 * 1024,
            "process_count": 4096,
            "rss_bytes": 6 * 1024 * 1024 * 1024,
            "enforced_by": "/usr/bin/prlimit",
            "address_space_enforcement": "kernel-prlimit-v1",
            "process_count_enforcement": "kernel-prlimit-v1",
            "rss_enforcement": "parent-sampling-only-v1",
        }:
            raise EvidenceError("exact-action compiler kernel/RSS limits are not canonical")
        identity = result.get("exec_identity")
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"pid", "starttime", "ppid", "pgrp", "exe_dev", "exe_ino", "sealed_dev", "sealed_ino", "exe_path", "argv_sha256", "cwd", "exec_ready"}
            or identity.get("pid") != result["pid"]
            or identity.get("starttime") != result["starttime"]
            or identity.get("ppid") != result["ppid"]
            or identity.get("pgrp") != result["pgrp"]
            or result["pgrp"] != result["pid"]
            or result["ppid"] == result["pid"]
            or identity.get("argv_sha256") != sha256_json(sealed_argv)
            or identity.get("cwd") != manifest["cwd"]["path"]
            or identity.get("exe_dev") != identity.get("sealed_dev")
            or identity.get("exe_ino") != identity.get("sealed_ino")
            or identity.get("exe_path") != f"/proc/{result['pid']}/exe"
            or identity.get("exec_ready") is not True
        ):
            raise EvidenceError("exact-action sealed executable/argv0/cwd proof is invalid")
        normalized.append(dict(event))
    if consumed != {action_id for action_id, item in issued.items() if item["state"] == "consumed"}:
        raise EvidenceError("exact-action transcript consumption is not one-to-one")
    if consumed != set(issued) or len(events) != len(expected_recipe_keys):
        raise EvidenceError("semantic G1 exact-action transcript leaves a reviewed action unconsumed or unacknowledged")
    closure = compiler.get("closure")
    if (
        not isinstance(closure, Mapping)
        or set(closure) != {"state", "build_root_pid", "build_root_starttime", "build_root_pgrp", "build_tree_reaped", "listener_closed", "active_requests", "quiescence_rounds", "state_machine", "request_count", "last_sequence", "events_sha256"}
        or closure.get("state") != "closed"
        or closure.get("build_tree_reaped") is not True
        or closure.get("listener_closed") is not True
        or closure.get("active_requests") != 0
        or closure.get("quiescence_rounds") != 3
        or closure.get("state_machine") != "new-running-closing-closed-v1"
        or any(isinstance(closure.get(name), bool) or not isinstance(closure.get(name), int) or closure[name] < 1 for name in ("build_root_pid", "build_root_starttime", "build_root_pgrp"))
        or closure.get("request_count") != len(events)
        or closure.get("last_sequence") != len(events) - 1
        or closure.get("events_sha256") != sha256_json(normalized)
    ):
        raise EvidenceError("exact-action closure proof does not match the complete transcript")
    if compiler.get("events_sha256") != sha256_json(normalized):
        raise EvidenceError("exact-action transcript digest is not exact")
    return dict(compiler)


def validate_compiler_execution_record(
    compiler: Mapping[str, Any], *, live_nonce: str | None = None, live_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject caller-supplied compiler liveness claims unconditionally.

    Serialized data is useful to audit after controller emission, but no
    importable validator can turn caller-provided nonce/event objects into the
    parent observation that authenticated and inspected the broker transcript.  The controller
    builder validates the closed serialization privately only after consuming
    its one-shot live observation capability.
    """

    del compiler, live_nonce, live_events
    raise EvidenceError("an importable compiler transcript has no live parent-observed authority")


def _validate_metadata(metadata: Mapping[str, Any], *, row: Mapping[str, Any], identity: Mapping[str, Any], repo: Path, authority: Mapping[str, Any] | None = None) -> None:
    validate_schema(dict(metadata), _schema(repo, ARTIFACT_SCHEMA), "semantic G1 builder metadata")
    required = {"schema_version", "metadata_id", "row_id", "target", "candidate", "authority", "artifact_kind", "command", "scope", "codegen", "contracts", "records", "runtime_libraries", "runtime_dependency_closure", "compiler_execution"}
    if set(metadata) != required or metadata.get("schema_version") != "rmsnorm-semantic-g1-artifact-v1" or metadata.get("metadata_id") != f"rmsnorm-semantic-g1-artifact-{row['target']}":
        raise EvidenceError("builder metadata is not a closed semantic G1 artifact")
    if metadata.get("row_id") != row["row_id"] or metadata.get("target") != row["target"] or metadata.get("artifact_kind") != "rmsnorm-semantic-g1-runtime":
        raise EvidenceError("builder metadata row/target/kind mismatch")
    _validate_candidate_document(metadata["candidate"], identity, "builder metadata")
    if authority is None or metadata.get("authority") != _require_live_authority(authority, identity, repo, "builder metadata"):
        raise EvidenceError("builder metadata authority does not equal the live reviewed controller authority")
    if metadata.get("command") != EXPECTED_COMMAND or metadata.get("scope") != EXPECTED_SCOPE or metadata.get("codegen") != EXPECTED_CODEGEN[row["target"]] or metadata.get("contracts") != authority_contract_hashes(authority):
        raise EvidenceError("builder metadata command/scope/codegen/contracts mismatch")
    records = metadata.get("records")
    if not isinstance(records, Mapping) or set(records) != {"binary", "companion", "loader"}:
        raise EvidenceError("builder metadata records are malformed")
    for name in records:
        _validate_digest_record(records[name], f"builder metadata {name}")
    libraries = metadata.get("runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise EvidenceError("builder metadata runtime libraries are missing")
    for entry in libraries:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "record"}:
            raise EvidenceError("builder metadata runtime library record is malformed")
        _validate_digest_record(entry["record"], f"builder runtime library {entry['name']}")
    closure = metadata.get("runtime_dependency_closure")
    if not isinstance(closure, Mapping) or closure.get("complete") is not True:
        raise EvidenceError("builder metadata does not carry a complete runtime dependency closure")
    if not isinstance(closure.get("objects"), list) or not closure["objects"]:
        raise EvidenceError("builder metadata runtime dependency closure is empty")
    compiler = metadata.get("compiler_execution")
    if not isinstance(compiler, Mapping):
        raise EvidenceError("builder compiler execution proof is malformed")
    # Metadata is a retained artifact, not the live compiler authority.  The
    # controller compares it to the private BuildResult observation before
    # allowing it to influence a PASS.
    _validate_serialized_compiler_execution(compiler)


@dataclass
class RuntimeBundle:
    metadata: SealedDescriptor
    binary: SealedDescriptor
    companion: SealedDescriptor
    loader: SealedDescriptor
    libraries: tuple[SealedDescriptor, ...]
    sidecars: tuple[SealedDescriptor, SealedDescriptor, SealedDescriptor]
    metadata_document: dict[str, Any]

    def close(self) -> None:
        for descriptor in (*self.sidecars, *self.libraries, self.loader, self.companion, self.binary, self.metadata):
            descriptor.close()


def elf_interpreter_path(descriptor: int) -> Path | None:
    header = os.pread(descriptor, 64, 0)
    if len(header) < 64 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
        return None
    phoff = struct.unpack_from("<Q", header, 32)[0]
    phentsize = struct.unpack_from("<H", header, 54)[0]
    phnum = struct.unpack_from("<H", header, 56)[0]
    if phentsize < 56 or phnum > 128:
        raise EvidenceError("runtime executable program headers are malformed")
    for index in range(phnum):
        program = os.pread(descriptor, phentsize, phoff + index * phentsize)
        if len(program) < 56:
            raise EvidenceError("runtime executable program header is truncated")
        if struct.unpack_from("<I", program, 0)[0] != 3:
            continue
        offset = struct.unpack_from("<Q", program, 8)[0]
        length = struct.unpack_from("<Q", program, 32)[0]
        raw = os.pread(descriptor, length, offset)
        if length < 2 or length > 4096 or len(raw) != length or raw[-1:] != b"\0":
            raise EvidenceError("runtime executable PT_INTERP is invalid")
        try:
            value = raw[:-1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise EvidenceError("runtime executable PT_INTERP is not ASCII") from exc
        if not Path(value).is_absolute():
            raise EvidenceError("runtime executable PT_INTERP is not absolute")
        return Path(value)
    return None


def _elf_dynamic(data: bytes, label: str) -> dict[str, Any]:
    """Parse the small ELF subset needed for a closed PT_INTERP/DT_NEEDED proof."""

    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise EvidenceError(f"{label} is not a little-endian ELF64 object")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    if phentsize < 56 or phnum > 128 or phoff + phentsize * phnum > len(data):
        raise EvidenceError(f"{label} ELF program headers are malformed")
    programs: list[tuple[int, int, int, int, int]] = []
    for index in range(phnum):
        offset = phoff + index * phentsize
        p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz, _memsz, _align = struct.unpack_from("<IIQQQQQQ", data, offset)
        if p_offset + p_filesz > len(data):
            raise EvidenceError(f"{label} ELF program segment is outside the object")
        programs.append((p_type, p_offset, p_vaddr, p_filesz, p_offset + p_filesz))

    def vaddr_to_offset(value: int) -> int:
        for p_type, p_offset, p_vaddr, p_filesz, _end in programs:
            if p_type == 1 and p_vaddr <= value < p_vaddr + p_filesz:
                return p_offset + value - p_vaddr
        raise EvidenceError(f"{label} ELF virtual address is not file-backed")

    interpreter: str | None = None
    dynamic_offset: int | None = None
    dynamic_size = 0
    for p_type, p_offset, _p_vaddr, p_filesz, _end in programs:
        if p_type == 3:
            raw = data[p_offset:p_offset + p_filesz]
            if len(raw) < 2 or raw[-1:] != b"\0" or b"\0" in raw[:-1]:
                raise EvidenceError(f"{label} ELF PT_INTERP is malformed")
            try:
                interpreter = raw[:-1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise EvidenceError(f"{label} ELF PT_INTERP is not ASCII") from exc
            if not Path(interpreter).is_absolute():
                raise EvidenceError(f"{label} ELF PT_INTERP is not absolute")
        elif p_type == 2:
            dynamic_offset, dynamic_size = p_offset, p_filesz
    needed: list[str] = []
    rpath: list[str] | None = None
    runpath: list[str] | None = None
    flags_1 = 0
    if dynamic_offset is not None:
        if dynamic_size < 16 or dynamic_offset + dynamic_size > len(data):
            raise EvidenceError(f"{label} ELF PT_DYNAMIC is malformed")
        entries: list[tuple[int, int]] = []
        for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
            tag, value = struct.unpack_from("<qQ", data, offset)
            entries.append((tag, value))
            if tag == 0:
                break
        strtab_value = next((value for tag, value in entries if tag == 5), None)
        strsz = next((value for tag, value in entries if tag == 10), None)
        if strtab_value is not None and strsz is not None:
            strtab_offset = vaddr_to_offset(strtab_value)
            if strsz < 1 or strtab_offset + strsz > len(data):
                raise EvidenceError(f"{label} ELF string table is malformed")
            strings = data[strtab_offset:strtab_offset + strsz]

            def string_at(value: int) -> str:
                if value >= len(strings):
                    raise EvidenceError(f"{label} ELF string offset is outside DT_STRTAB")
                end = strings.find(b"\0", value)
                if end < 0:
                    raise EvidenceError(f"{label} ELF string is unterminated")
                try:
                    return strings[value:end].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise EvidenceError(f"{label} ELF dependency name is not UTF-8") from exc

            for tag, value in entries:
                if tag == 1:
                    name = string_at(value)
                    if not name or "/" in name:
                        raise EvidenceError(f"{label} ELF DT_NEEDED name is unsafe")
                    needed.append(name)
                elif tag == 15:
                    rpath = string_at(value).split(":")
                elif tag == 29:
                    runpath = string_at(value).split(":")
                elif tag == 0x6FFFFFFB:
                    flags_1 = int(value)
    return {
        "interpreter": interpreter,
        "needed": needed,
        "rpath": rpath,
        "runpath": runpath,
        "flags_1": flags_1,
    }


_DEFAULT_LOADER_DIRS = (
    "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib64", "/usr/lib64", "/lib", "/usr/lib",
)
_HWCAPS = ("x86-64-v4", "x86-64-v3", "x86-64-v2")
_DF_1_NODEFLIB = 0x800


def _split_loader_path_list(entries: Sequence[str] | None, owner: Path, label: str) -> list[str]:
    if entries is None:
        return []
    result: list[str] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            # glibc treats an empty element as the current directory.  The
            # current directory is not part of a sealed artifact authority;
            # fail closed rather than silently proving a different directory.
            raise EvidenceError(f"{label} contains an empty loader path element")
        expanded = entry
        for token, value in (
            ("$ORIGIN", str(owner.parent)),
            ("${ORIGIN}", str(owner.parent)),
            ("$LIB", "lib64"),
            ("${LIB}", "lib64"),
            ("$PLATFORM", "x86_64"),
            ("${PLATFORM}", "x86_64"),
        ):
            expanded = expanded.replace(token, value)
        if "$" in expanded:
            raise EvidenceError(f"{label} contains an unsupported loader token")
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        result.append(str(candidate))
    return result


def _ldconfig_cache() -> dict[str, list[str]]:
    try:
        completed = subprocess.run(
            ["/sbin/ldconfig", "-p"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("glibc loader cache could not be captured") from exc
    cache: dict[str, list[str]] = {}
    for raw_line in completed.stdout.decode("utf-8", "strict").splitlines():
        if " => " not in raw_line or not raw_line.startswith(" "):
            continue
        left, right = raw_line.strip().split(" => ", 1)
        name = left.split(" (", 1)[0]
        path = Path(right)
        if not name or not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_file() or resolved.is_symlink():
            continue
        cache.setdefault(name, []).append(str(resolved))
    for name in cache:
        cache[name] = list(dict.fromkeys(cache[name]))
    return cache


def _loader_context(cache: Mapping[str, Sequence[str]], names: Sequence[str]) -> dict[str, Any]:
    return {
        "ld_library_path": SANITIZED_RUNTIME_LD_LIBRARY_PATH.split(":"),
        "platform": "x86_64",
        "lib": "lib64",
        "default_dirs": list(_DEFAULT_LOADER_DIRS),
        "hwcaps": list(_HWCAPS),
        "cache": {name: list(cache.get(name, ())) for name in sorted(set(names))},
    }


def _dependency_search_paths(
    owner: Path,
    dynamic: Mapping[str, Any],
    inherited_rpath: Sequence[str],
    cache: Mapping[str, Sequence[str]],
    name: str,
) -> list[tuple[Path, str]]:
    """Model glibc's RPATH/RUNPATH, environment, cache and hwcaps order."""

    paths: list[tuple[Path, str]] = []
    own_rpath = _split_loader_path_list(dynamic.get("rpath"), owner, f"{owner} DT_RPATH")
    own_runpath = _split_loader_path_list(dynamic.get("runpath"), owner, f"{owner} DT_RUNPATH")
    if dynamic.get("runpath") is None:
        paths.extend((Path(value), "rpath") for value in (*inherited_rpath, *own_rpath))
    else:
        paths.extend((Path(value), "rpath") for value in inherited_rpath)
    paths.extend((Path(value), "ld-library-path") for value in SANITIZED_RUNTIME_LD_LIBRARY_PATH.split(":"))
    if dynamic.get("runpath") is not None:
        paths.extend((Path(value), "runpath") for value in own_runpath)
    if not (int(dynamic.get("flags_1", 0)) & _DF_1_NODEFLIB):
        for cached in cache.get(name, ()):
            paths.append((Path(cached).parent, "cache"))
        for value in _DEFAULT_LOADER_DIRS:
            base = Path(value)
            paths.extend((base / "glibc-hwcaps" / hwcap, "default-hwcaps") for hwcap in _HWCAPS)
            paths.append((base, "default"))
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, kind in paths:
        try:
            normalized = path.resolve(strict=False)
        except OSError as exc:
            raise EvidenceError(f"glibc loader search path could not be normalized: {path}") from exc
        if normalized not in seen:
            seen.add(normalized)
            result.append((normalized, kind))
    return result


def runtime_dependency_closure(path: Path) -> dict[str, Any]:
    """Capture a recursive glibc loader proof from live regular files."""

    root = path.resolve(strict=True)
    cache = _ldconfig_cache()
    queue: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    objects: dict[str, dict[str, Any]] = {}
    all_names: list[str] = []
    while queue:
        current, inherited_rpath = queue.pop(0)
        current = current.resolve(strict=True)
        key = str(current)
        if key in objects:
            continue
        descriptor = _open_regular(current, "runtime dependency")
        try:
            data = fd_read_all(descriptor, max_bytes=256 * 1024 * 1024)
            record = _record_from_descriptor(descriptor, path=current)
        finally:
            os.close(descriptor)
        dynamic = _elf_dynamic(data, str(current))
        interpreter_path = None if dynamic["interpreter"] is None else Path(dynamic["interpreter"]).resolve(strict=True)
        if interpreter_path is not None:
            queue.append((interpreter_path, ()))
        edges: list[dict[str, str]] = []
        next_inherited = tuple(inherited_rpath)
        if dynamic.get("runpath") is None:
            next_inherited = tuple((*inherited_rpath, *_split_loader_path_list(dynamic.get("rpath"), current, f"{current} DT_RPATH")))
        for name in dynamic["needed"]:
            all_names.append(name)
            resolved = None
            selected_kind = None
            if "/" in name:
                candidate = Path(name)
                if not candidate.is_absolute():
                    raise EvidenceError(f"runtime dependency path is not absolute: {name}")
                try:
                    candidate_resolved = candidate.resolve(strict=True)
                    if candidate_resolved.is_file() and not candidate_resolved.is_symlink():
                        resolved = candidate_resolved
                except OSError:
                    resolved = None
            else:
                for directory, kind in _dependency_search_paths(current, dynamic, inherited_rpath, cache, name):
                    candidate = directory / name
                    try:
                        candidate_resolved = candidate.resolve(strict=True)
                        if candidate_resolved.is_file() and not candidate_resolved.is_symlink():
                            resolved = candidate_resolved
                            selected_kind = kind
                            break
                    except OSError:
                        continue
            if resolved is None:
                raise EvidenceError(f"runtime dependency {name} is missing for {current}")
            if selected_kind is None:
                selected_kind = "direct"
            edges.append({"name": name, "resolved_path": str(resolved), "search_kind": selected_kind})
            queue.append((resolved, next_inherited))
        objects[key] = {
            "record": record,
            "interpreter": None if interpreter_path is None else str(interpreter_path),
            "needed": edges,
            "rpath": dynamic.get("rpath"),
            "runpath": dynamic.get("runpath"),
            "flags_1": int(dynamic.get("flags_1", 0)),
            "inherited_rpath": list(inherited_rpath),
        }
    ordered = [objects[key] for key in sorted(objects)]
    context = _loader_context(cache, all_names)
    closure = {
        "complete": True,
        "algorithm": "glibc-loader-closure-v2",
        "root": str(root),
        "loader_context": context,
        "objects": ordered,
    }
    closure["sha256"] = sha256_json(closure)
    return closure


def validate_runtime_dependency_closure(
    retained: Mapping[str, tuple[Mapping[str, Any], int]],
    closure: Mapping[str, Any],
    *,
    root_path: str,
    loader_path: str,
) -> None:
    """Re-prove the closure from sealed bytes at the controller boundary."""

    if set(closure) != {"complete", "algorithm", "root", "loader_context", "objects", "sha256"} or closure.get("complete") is not True or closure.get("algorithm") != "glibc-loader-closure-v2":
        raise EvidenceError("runtime dependency closure is incomplete or uses an unknown proof algorithm")
    unsigned = {key: closure[key] for key in ("complete", "algorithm", "root", "loader_context", "objects")}
    if closure.get("sha256") != sha256_json(unsigned):
        raise EvidenceError("runtime dependency closure digest is not canonical")
    context = closure.get("loader_context")
    if not isinstance(context, Mapping) or set(context) != {"ld_library_path", "platform", "lib", "default_dirs", "hwcaps", "cache"} or context.get("ld_library_path") != SANITIZED_RUNTIME_LD_LIBRARY_PATH.split(":") or context.get("platform") != "x86_64" or context.get("lib") != "lib64" or context.get("default_dirs") != list(_DEFAULT_LOADER_DIRS) or context.get("hwcaps") != list(_HWCAPS) or not isinstance(context.get("cache"), Mapping):
        raise EvidenceError("runtime dependency loader context is not the canonical live glibc context")
    live_cache = _ldconfig_cache()
    if {str(name): list(values) for name, values in context["cache"].items()} != {str(name): list(live_cache.get(str(name), ())) for name in context["cache"]}:
        raise EvidenceError("runtime dependency loader cache changed after the builder proof")
    objects = closure.get("objects")
    if not isinstance(objects, list) or not objects:
        raise EvidenceError("runtime dependency closure object set is empty")
    object_map: dict[str, Mapping[str, Any]] = {}
    dynamic_map: dict[str, dict[str, Any]] = {}
    for item in objects:
        if not isinstance(item, Mapping) or not isinstance(item.get("record"), Mapping):
            raise EvidenceError("runtime dependency closure object record is malformed")
        record = _validate_digest_record(item["record"], "runtime dependency closure object")
        key = str(record["resolved_path"])
        if key in object_map:
            raise EvidenceError("runtime dependency closure contains duplicate object paths")
        object_map[key] = item
    for item in objects:
        if not isinstance(item, Mapping) or set(item) != {"record", "interpreter", "needed", "rpath", "runpath", "flags_1", "inherited_rpath"} or not isinstance(item.get("record"), Mapping):
            raise EvidenceError("runtime dependency closure object record is malformed")
        record = _validate_digest_record(item["record"], "runtime dependency closure object")
        key = str(record["resolved_path"])
        retained_record = retained.get(key)
        if retained_record is None or retained_record[0].get("sha256") != record["sha256"] or fd_sha256(retained_record[1]) != record["sha256"]:
            raise EvidenceError("runtime dependency closure object bytes were missing or replaced")
        data = fd_read_all(retained_record[1], max_bytes=256 * 1024 * 1024)
        dynamic = _elf_dynamic(data, key)
        observed_interpreter = None
        if dynamic["interpreter"] is not None:
            literal_interpreter = str(dynamic["interpreter"])
            matched_interpreter = next(
                (
                    retained_entry[0]
                    for retained_entry in retained.values()
                    if retained_entry[0].get("path") == literal_interpreter
                    or retained_entry[0].get("resolved_path") == literal_interpreter
                ),
                None,
            )
            if matched_interpreter is None:
                raise EvidenceError("runtime dependency PT_INTERP does not identify a retained sealed object")
            observed_interpreter = str(matched_interpreter["resolved_path"])
        if item.get("interpreter") != observed_interpreter:
            raise EvidenceError("runtime dependency PT_INTERP proof differs from the sealed record")
        if item.get("rpath") != dynamic.get("rpath") or item.get("runpath") != dynamic.get("runpath") or item.get("flags_1") != dynamic.get("flags_1"):
            raise EvidenceError("runtime dependency RPATH/RUNPATH or loader flags differ from sealed bytes")
        inherited = item.get("inherited_rpath")
        if not isinstance(inherited, list) or any(not isinstance(value, str) or not Path(value).is_absolute() for value in inherited):
            raise EvidenceError("runtime dependency inherited RPATH proof is malformed")
        dynamic_map[key] = dynamic
        needed = item.get("needed")
        if not isinstance(needed, list) or any(not isinstance(edge, Mapping) or set(edge) != {"name", "resolved_path", "search_kind"} for edge in needed):
            raise EvidenceError("runtime dependency edge record is malformed")
        if [edge["name"] for edge in needed] != dynamic["needed"]:
            raise EvidenceError("runtime dependency DT_NEEDED names differ from sealed bytes")
        for edge in needed:
            if not isinstance(edge["search_kind"], str) or edge["search_kind"] not in {"direct", "rpath", "ld-library-path", "runpath", "cache", "default-hwcaps", "default"}:
                raise EvidenceError("runtime dependency search order proof is malformed")
            if str(edge["resolved_path"]) not in object_map:
                raise EvidenceError("runtime dependency closure omits a transitive dependency")
    root = str(closure["root"])
    if root != root_path or root not in object_map or object_map[root].get("interpreter") != loader_path:
        raise EvidenceError("runtime executable PT_INTERP does not bind the retained loader")
    expected_inherited: dict[str, tuple[str, ...]] = {root: ()}
    search_pending = [root]
    while search_pending:
        owner_key = search_pending.pop(0)
        owner = object_map[owner_key]
        inherited = expected_inherited[owner_key]
        if tuple(owner.get("inherited_rpath", ())) != inherited:
            raise EvidenceError("runtime dependency inherited RPATH is not the glibc traversal value")
        dynamic = dynamic_map[owner_key]
        next_inherited = inherited
        if dynamic.get("runpath") is None:
            next_inherited = tuple((*inherited, *_split_loader_path_list(dynamic.get("rpath"), Path(owner_key), f"{owner_key} DT_RPATH")))
        interpreter = owner.get("interpreter")
        if interpreter is not None and str(interpreter) not in expected_inherited:
            expected_inherited[str(interpreter)] = ()
            search_pending.append(str(interpreter))
        for edge in owner["needed"]:
            name = str(edge["name"])
            selected: tuple[str, str] | None = None
            for directory, kind in _dependency_search_paths(Path(owner_key), dynamic, inherited, context["cache"], name):
                candidate = directory / name
                try:
                    resolved_candidate = candidate.resolve(strict=True)
                except OSError:
                    continue
                if resolved_candidate.is_file() and not resolved_candidate.is_symlink():
                    selected = (str(resolved_candidate), kind)
                    break
            if selected is None or selected != (str(edge["resolved_path"]), str(edge["search_kind"])):
                raise EvidenceError("runtime dependency edge does not match glibc search order and live object bytes")
            child_key = str(edge["resolved_path"])
            if child_key not in expected_inherited:
                expected_inherited[child_key] = next_inherited
                search_pending.append(child_key)
    if set(expected_inherited) != set(object_map):
        raise EvidenceError("runtime dependency closure contains unreachable or unmodeled loader objects")
    reachable: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        item = object_map.get(current)
        if item is None:
            raise EvidenceError("runtime dependency closure traversal found a missing object")
        reachable.add(current)
        if item.get("interpreter") is not None:
            pending.append(str(item["interpreter"]))
        pending.extend(str(edge["resolved_path"]) for edge in item["needed"])
    if reachable != set(object_map):
        raise EvidenceError("runtime dependency closure contains unreachable or omitted transitive objects")


def capture_builder_bundle(result: Any, *, row: Mapping[str, Any], identity: Mapping[str, Any], repo: Path, authority: Mapping[str, Any]) -> RuntimeBundle:
    """Freeze builder output at the controller boundary before any use.

    No later launch, parse, sidecar check, or aggregate input rereads a builder
    pathname.  The build result's direct return digest is the authority for
    metadata; metadata is not allowed to self-authenticate itself.
    """

    if getattr(result, "runtime_dependency_closure_complete", False) is not True:
        raise EvidenceError("FAIL-CLOSED: complete runtime DT_NEEDED dependency closure was not independently captured")
    output_root = Path(result.output_dir).resolve(strict=True)
    metadata_path = Path(result.metadata_path)
    try:
        if metadata_path.resolve(strict=True).parent != output_root:
            raise EvidenceError("builder metadata does not remain inside its private row directory")
    except OSError as exc:
        raise EvidenceError("builder metadata path cannot be resolved") from exc
    metadata_expected = {"path": str(metadata_path), "resolved_path": str(metadata_path.resolve(strict=True)), "size_bytes": metadata_path.stat().st_size, "sha256": str(result.metadata_sha256)}
    metadata = snapshot_file(metadata_path, metadata_expected, "builder metadata")
    created: list[SealedDescriptor] = [metadata]
    try:
        document = read_json_bytes(fd_read_all(metadata.fd, max_bytes=MAX_OUTPUT * 4), "builder metadata")
        if not isinstance(document, dict):
            raise EvidenceError("builder metadata JSON is not an object")
        live_compiler = getattr(result, "compiler_execution", None)
        if not isinstance(live_compiler, Mapping) or document.get("compiler_execution") != dict(live_compiler):
            raise EvidenceError("builder metadata compiler transcript does not equal the direct parent-observed BuildResult")
        _validate_metadata(document, row=row, identity=identity, repo=repo, authority=authority)
        records = document["records"]
        expected_local = {
            "binary": output_root / BINARY_NAME,
            "companion": output_root / COMPANION_NAME.format(target=row["target"]),
        }
        for name, expected_path in expected_local.items():
            try:
                observed_path = Path(records[name]["path"]).resolve(strict=True)
            except OSError as exc:
                raise EvidenceError(f"builder {name} path cannot be resolved") from exc
            if observed_path != expected_path:
                raise EvidenceError(f"builder {name} record escapes its target-qualified row directory")
        binary = snapshot_file(Path(records["binary"]["path"]), records["binary"], "builder runtime binary")
        companion = snapshot_file(Path(records["companion"]["path"]), records["companion"], "builder device companion")
        loader = snapshot_file(Path(records["loader"]["path"]), records["loader"], "builder dynamic loader")
        created.extend((binary, companion, loader))
        libraries = tuple(snapshot_file(Path(entry["record"]["path"]), entry["record"], f"builder runtime library {entry['name']}") for entry in document["runtime_libraries"])
        created.extend(libraries)
        interpreter_path = elf_interpreter_path(binary.fd)
        if not libraries or interpreter_path is None:
            raise EvidenceError("sealed runtime executable does not bind a dynamic loader")
        if str(interpreter_path) != str(records["loader"]["path"]):
            raise EvidenceError("sealed runtime executable does not bind the sealed loader")
        retained = {
            str(binary.record["resolved_path"]): (binary.record, binary.fd),
            str(loader.record["resolved_path"]): (loader.record, loader.fd),
            **{str(item.record["resolved_path"]): (item.record, item.fd) for item in libraries},
        }
        validate_runtime_dependency_closure(
            retained,
            document["runtime_dependency_closure"],
            root_path=str(binary.record["resolved_path"]),
            loader_path=str(loader.record["resolved_path"]),
        )
        sidecars = (
            snapshot_file(metadata_path.with_name(metadata_path.name + SIDECAR_SUFFIX), None, "builder metadata sidecar"),
            snapshot_file(Path(records["binary"]["path"]).with_name(BINARY_NAME + SIDECAR_SUFFIX), None, "builder binary sidecar"),
            snapshot_file(Path(records["companion"]["path"]).with_name(COMPANION_NAME.format(target=row["target"]) + SIDECAR_SUFFIX), None, "builder companion sidecar"),
        )
        created.extend(sidecars)
        validate_sidecar(sidecars[0], target_record=metadata.record, filename=METADATA_NAME, label="builder metadata sidecar")
        validate_sidecar(sidecars[1], target_record=records["binary"], filename=BINARY_NAME, label="builder binary sidecar")
        validate_sidecar(sidecars[2], target_record=records["companion"], filename=COMPANION_NAME.format(target=row["target"]), label="builder companion sidecar")
        if binary.record["sha256"] != str(result.artifact_sha256) or companion.record["sha256"] != str(result.companion_sha256):
            raise EvidenceError("builder result descriptor disagrees with controller-captured bytes")
        return RuntimeBundle(metadata, binary, companion, loader, libraries, sidecars, document)
    except BaseException:
        for descriptor in reversed(created):
            descriptor.close()
        raise


def validate_workflow_registration(repo: Path = ROOT) -> None:
    repo = canonical_repository(repo)
    try:
        path = repo / WORKFLOW_PATH
        data = _reviewed_contract_bytes(repo, WORKFLOW_PATH)
        if data is None:
            data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError("canonical semantic G1 GPU workflow is missing") from exc
    if sha256_bytes(data) != SEMANTIC_G1_WORKFLOW_SHA256:
        raise EvidenceError("canonical semantic G1 workflow bytes differ from the complete reviewed workflow")
    required = (
        "exec /usr/bin/env -i",
        "PATH=/usr/bin:/bin",
        "CI=true",
        "GITHUB_ACTIONS=true",
        "GITHUB_WORKFLOW=semantic-rmsnorm-g1",
        "--no-replace-objects",
        "/usr/bin/python3 -I -S -c '",
        "os.memfd_create(\"sllm-semantic-g1-controller\"",
        "SLLM_G1_CONTROLLER_FD",
        '"/proc/self/fd/{fd}"',
        "Upload controller-validated evidence",
        "Remove private controller directories",
    )
    upload_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("${{ env.RUN_ROOT }}/")]
    if "--repo" in text or "PYTHONPATH" in text or any(fragment not in text for fragment in required) or any(fragment not in text for fragment in SEMANTIC_G1_UPLOAD_PATHS) or any("**" in line for line in upload_lines):
        raise EvidenceError("canonical semantic G1 workflow does not use the fixed isolated controller boundary")
    if upload_lines != list(SEMANTIC_G1_UPLOAD_PATHS):
        raise EvidenceError("canonical semantic G1 workflow upload paths are not the explicit reviewed allowlist")
    if any("*" in line or "?" in line or any(marker in line.lower() for marker in ("trace", "slice", "weight", "device-code-object", "sllm-rmsnorm-g1-evidence", ".elf", ".so")) for line in upload_lines):
        raise EvidenceError("canonical semantic G1 workflow upload allowlist contains a runtime/raw artifact")


def validate_compiler_execution_contract(repo: Path = ROOT) -> None:
    """Require the parent-owned broker protocol at every build boundary."""

    repo = canonical_repository(repo)
    try:
        builder_source = (repo / "ci/tools/build_rmsnorm_g1_runtime.py").read_text(encoding="utf-8")
        build_source = (repo / RUST_BUILD_RELATIVE_PATH).read_text(encoding="utf-8")
        cmake_source = (repo / HIP_CMAKE_RELATIVE_PATH).read_text(encoding="utf-8")
        orchestrator_source = (repo / "ci/tools/orchestrate_rmsnorm_g1_evidence.py").read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError("semantic G1 compiler binding sources are unavailable") from exc
    required_builder = (
        "COMPILER_BROKER_AVAILABLE = True",
        "class CompilerBroker",
        "socket.AF_UNIX",
        "socket.SOCK_SEQPACKET",
        "socket.SO_PASSCRED",
        "execveat",
        "COMPILER_BROKER_SOCKET_ENV",
        "COMPILER_BROKER_TOKEN_ENV",
        "COMPILER_BROKER_SESSION_ENV",
        "request_nonce",
        "exact_actions.OneShotBroker",
        "action_manifest",
        "action_digest",
        "build_tree_reaped",
        "os.posix_spawn(",
        "os.pidfd_open",
        "signal.pidfd_send_signal",
        "COMPILER_EXEC_HELPER_SOURCE",
        "exec_ready",
        "parent-issued-exact-action-v1",
        "COMPILER_BROKER_CLIENT_FD_ENV",
        "process_limiter_snapshot",
        "parent-sampling-only-v1",
        "_launch_slot",
        "_close_rights",
        "rss_enforcement",
        "seal_input_view",
        "sealed_input_view",
        "spawn_file_actions",
        "validate_live_manifest",
    )
    required_build = (
        "struct PinnedCompiler",
        "SLLM_SEMANTIC_G1_AUTHORITY",
        "SLLM_HIP_COMPILER_BROKER_CLIENT",
        "SLLM_HIP_COMPILER_BROKER_SOCKET",
        "SLLM_HIP_COMPILER_BROKER_TOKEN",
        "SLLM_HIP_COMPILER_BROKER_SESSION",
        "client_path()",
        "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256",
        "SLLM_HIP_COMPILER_BROKER_CLIENT_FD",
        "CMAKE_HIP_COMPILER",
    )
    required_cmake = (
        "SLLM_SEMANTIC_G1_AUTHORITY",
        "SLLM_HIP_COMPILER_LOGICAL",
        "SLLM_HIP_COMPILER_BROKER_CLIENT",
        "SLLM_HIP_COMPILER_BROKER_SOCKET",
        "SLLM_HIP_COMPILER_BROKER_SESSION",
        "SLLM_HIP_COMPILER_BROKER_TOKEN",
        "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256",
        "SLLM_HIP_COMPILER_BROKER_CLIENT_FD",
        "/proc/self/fd/",
        "execute_process(",
    )
    if any(fragment not in builder_source for fragment in required_builder):
        raise EvidenceError("semantic G1 builder does not implement the reviewed parent-owned broker")
    if any(fragment not in build_source for fragment in required_build) or any(fragment not in cmake_source for fragment in required_cmake):
        raise EvidenceError("semantic G1 build boundaries are not bound to the authenticated broker client")
    if "ci/tools/exact_actions.py" not in orchestrator_source or '_load_reviewed_module("exact_actions", "ci/tools/exact_actions.py")' not in orchestrator_source:
        raise EvidenceError("semantic G1 controller does not load the reviewed exact-action module")
    forbidden_builder = (
        "COMPILER_BROKER_AVAILABLE = False",
        "sealed-memfd-wrapper-events-v3",
        "EXPECTED_COMPILER_EVENT_PLAN",
        "compiler_fd",
        "wrapper_fd",
        "os.fork()",
        "resource.setrlimit",
        "ctypes.CDLL",
    )
    forbidden_build = (
        "SLLM_HIP_COMPILER_FD",
        "SLLM_HIP_COMPILER_WRAPPER_FD",
        "SLLM_HIP_COMPILER_EVENT_FD",
        "SLLM_HIP_COMPILER_EVENT_ACK_FD",
        "SLLM_HIP_COMPILER_EVENT_NONCE",
        "sealed-memfd-wrapper-events-v3",
        "compiler_wrapper",
    )
    forbidden_cmake = (
        "SLLM_HIP_COMPILER_FD",
        "SLLM_HIP_COMPILER_WRAPPER_FD",
        "SLLM_HIP_COMPILER_EVENT_FD",
        "SLLM_HIP_COMPILER_EVENT_ACK_FD",
        "SLLM_HIP_COMPILER_EVENT_NONCE",
        "sealed-memfd-wrapper-events-v3",
    )
    if any(fragment in builder_source for fragment in forbidden_builder) or any(fragment in build_source for fragment in forbidden_build) or any(fragment in cmake_source for fragment in forbidden_cmake):
        raise EvidenceError("semantic G1 still exposes the old compiler FD/wrapper event mechanism")
    forbidden_direct = (
        "write_compiler_execution_record",
        "executed_count\":1",
        "Command::new(\"cmake\")",
        "Command::new(\"rustc\")",
        "unwrap_or_else(|| \"c++\".into())",
    )
    if any(fragment in build_source for fragment in forbidden_direct) or "Command::new(&compiler)" in build_source:
        raise EvidenceError("semantic G1 still executes the mutable compiler pathname directly")


def validate_contracts(repo: Path = ROOT) -> None:
    repo = canonical_repository(repo)
    validate_matrix(repo)
    for relative in (ARTIFACT_SCHEMA, REPORT_SCHEMA, AGGREGATE_SCHEMA):
        _schema(repo, relative)
    validate_compiler_execution_contract(repo)
    validate_workflow_registration(repo)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_contracts(canonical_repository(args.repo))
    except (EvidenceError, ContractError, OSError, ValueError) as exc:
        print(f"semantic RMSNorm G1 contracts: FAIL: {exc}", file=sys.stderr)
        return 1
    print("semantic RMSNorm G1 contracts: PASS (controller-owned raw-evidence boundary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
