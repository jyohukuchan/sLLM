#!/usr/bin/env python3
"""Fail-closed contracts for real-weight RMSNorm G2.

The checked-in tests use only temporary synthetic safetensors fixtures.  The
runtime path additionally verifies a complete read-only model cache and keeps
the verified shard descriptor open while taking the bounded slice, so a path
replacement cannot turn into a different tensor between verification and use.
GPU execution remains delegated to the dedicated G2 binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from common import ContractError, ROOT, canonical_bytes, read_json, sha256_file, sha256_json  # noqa: E402

TARGETS = ("gfx1030", "gfx1201")
ROWS = tuple(f"rmsnorm-g2-{target}" for target in TARGETS)
CASE_IDS = (
    "g2-r1-n2560", "g2-r2-n2560", "g2-r17-n2560",
    "g2-r255-n2560", "g2-r256-n2560", "g2-r257-n2560",
)
CASE_ROWS = (1, 2, 17, 255, 256, 257)
CASE_SEEDS = (9201, 9202, 9217, 9255, 9256, 9257)
MODEL_LOCK_PATH = "docs/models/locks/qwen3.5-4b-bf16.json"
MODEL_LOCK_FINGERPRINT = "sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935"
RESOLVED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TENSOR_NAME = "model.language_model.layers.0.input_layernorm.weight"
SOURCE_SHARD = "model.safetensors-00002-of-00002.safetensors"
DATA_OFFSETS = (15360, 20480)
ABSOLUTE_RANGE = (94432, 99552)
HEADER_LENGTH = 79064
DATA_BUFFER_START = 79072
BYTE_SIZE = 5120
TOLERANCE_ID = "rmsnorm-bf16-f32-output-v1"
ATOL = 0.0078125
RTOL = 0.015625
G2_BINARY = "sllm-rmsnorm-g2-evidence"
G2_SIDECAR = G2_BINARY + ".sha256"
G2_SOURCE_PATH = "crates/sllm-hip/src/bin/sllm-rmsnorm-g2-evidence.rs"
G2_BUILD_INPUTS_PATH = "ci/matrix/rmsnorm-g2-build-inputs-v1.json"
G2_IDENTITY_SCHEMA = "rmsnorm-g2-build-identity-v1"
G2_IDENTITY_MARKER = b"SLLM_G2_BUILD_IDENTITY_V1:"
G2_ROLE = "dedicated-g2-runtime"
G2_BUILD_COMMAND = (
    "cargo", "+1.97.1", "build", "--locked", "--offline", "--release", "--bin", G2_BINARY,
)
G2_BUILD_PROFILE = "release"
G2_BUILDER_OUTPUT_PATH = f"target/{G2_BUILD_PROFILE}/{G2_BINARY}"
G2_RUNTIME_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib"
G2_CODEGEN_FEATURES = "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0"
PREREQUISITE_KINDS = ("g0", "private_g1", "semantic_g1", "h3")
SYNTHETIC_MARKER = "rmsnorm-g2-synthetic-safetensors-v1"
SCHEMAS = {
    "matrix": "ci/schema/rmsnorm-g2-matrix-v1.schema.json",
    "slice": "ci/schema/rmsnorm-g2-model-slice-v1.schema.json",
    "tolerance": "ci/schema/rmsnorm-g2-tolerance-v1.schema.json",
    "runtime_result": "ci/schema/rmsnorm-g2-runtime-result-v1.schema.json",
    "artifact": "ci/schema/rmsnorm-g2-artifact-v1.schema.json",
    "report": "ci/schema/rmsnorm-g2-report-v1.schema.json",
    "aggregate": "ci/schema/rmsnorm-g2-aggregate-v1.schema.json",
}
MATRIX_PATH = "ci/matrix/rmsnorm-g2-v1.json"
TOLERANCE_PATH = "ci/matrix/rmsnorm-g2-tolerance-v1.json"


def _schema(repo: Path, name: str) -> dict[str, Any]:
    value = read_json(repo / SCHEMAS[name])
    if not isinstance(value, dict):
        raise ContractError(f"G2 {name} schema is not an object")
    Draft202012Validator.check_schema(value)
    return value


def _schema_validate(repo: Path, name: str, document: Any) -> None:
    errors = sorted(
        Draft202012Validator(_schema(repo, name), format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        detail = "; ".join(f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors[:5])
        raise ContractError(f"G2 {name} schema rejected document: {detail}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha40(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or value.lower() != value or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{label} is not a complete lowercase SHA-1 identity")
    return value


def candidate_sha256(candidate: Mapping[str, Any]) -> str:
    return sha256_json(dict(candidate))


def validate_candidate(
    candidate: Mapping[str, Any],
    repo: Path = ROOT,
    *,
    strict_git: bool = False,
) -> dict[str, Any]:
    """Validate the one immutable candidate shape shared by every G2 record.

    Contract/unit tests may use an isolated synthetic candidate.  The command
    entry points pass ``strict_git=True`` so evidence cannot name a different
    commit or tree than the checkout that produced it.
    """

    required = {"reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "worktree_clean", "revision_input"}
    if set(candidate) != required:
        raise ContractError("G2 candidate identity has missing or unknown fields")
    values = []
    for name in ("reviewed_sha", "tested_sha", "workflow_sha"):
        value = candidate[name]
        if not isinstance(value, str) or len(value) != 40 or value.lower() != value or any(char not in "0123456789abcdef" for char in value) or value == "0" * 40:
            raise ContractError(f"G2 {name} is not a nonzero full lowercase commit SHA")
        values.append(value)
    tree = candidate["git_tree_oid"]
    if not isinstance(tree, str) or len(tree) != 40 or tree.lower() != tree or any(char not in "0123456789abcdef" for char in tree) or tree == "0" * 40:
        raise ContractError("G2 git_tree_oid is not a nonzero full lowercase tree OID")
    if len(set(values)) != 1:
        raise ContractError("G2 reviewed_sha, tested_sha, and workflow_sha must be identical")
    if candidate["worktree_clean"] is not True or candidate["revision_input"] != "full-sha":
        raise ContractError("G2 candidate is not a clean full-SHA identity")
    result = dict(candidate)
    if strict_git:
        def git(*args: str) -> str:
            completed = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise ContractError(f"cannot verify G2 Git identity: {completed.stderr.strip()}")
            return completed.stdout.strip()

        actual_commit = git("rev-parse", "--verify", "HEAD^{commit}")
        actual_tree = git("rev-parse", "--verify", f"{actual_commit}^{{tree}}")
        if actual_commit != values[0] or actual_tree != tree:
            raise ContractError("G2 candidate does not match the checked-out immutable commit/tree")
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            raise ContractError("G2 evidence requires a clean candidate worktree")
    return result


def _expected_target(target: str) -> dict[str, Any]:
    if target == "gfx1030":
        return {"bdf": "0000:03:00.0", "uuid": "GPU-76a08c022586fed6", "product": "AMD Radeon Pro V620", "physical_hip_index": 1, "logical_device_index": 0}
    if target == "gfx1201":
        return {"bdf": "0000:07:00.0", "uuid": "GPU-a8e9ddefa2d60f55", "product": "AMD Radeon AI PRO R9700", "physical_hip_index": 2, "logical_device_index": 0}
    raise ContractError(f"unsupported G2 target: {target}")


def expected_matrix() -> dict[str, Any]:
    return {
        "schema_version": "rmsnorm-g2-matrix-v1", "matrix_id": "rmsnorm-g2-v1",
        "suite_id": "g2-rmsnorm-real-weight-runtime", "tier": "tier_g2", "required": True,
        "targets": [
            {"order": order, "row_id": f"rmsnorm-g2-{target}", "target": target, "device": _expected_target(target), "backend": "hip", "cases": list(CASE_IDS)}
            for order, target in enumerate(TARGETS)
        ],
        "cases": [
            {"order": order, "id": case_id, "rows": rows, "n": 2560, "input_seed": seed, "classification": "finite"}
            for order, (case_id, rows, seed) in enumerate(zip(CASE_IDS, CASE_ROWS, CASE_SEEDS))
        ],
        "tolerance": TOLERANCE_PATH, "slice_contract": SCHEMAS["slice"], "result_schema": SCHEMAS["runtime_result"],
        "prerequisites": {
            "g0": "bound-only:g0-gfx1030-or-gfx1201",
            "private_g1": "bound-only:g1-gfx1030-or-gfx1201",
            "semantic_g1": "bound-only:rmsnorm-semantic-g1-gfx1030-or-gfx1201",
            "h3": "bound-only:h3-rmsnorm-gfx1030-or-gfx1201",
        },
    }


def validate_matrix(repo: Path = ROOT) -> dict[str, Any]:
    document = read_json(repo / MATRIX_PATH)
    _schema_validate(repo, "matrix", document)
    if document != expected_matrix():
        raise ContractError("G2 matrix target/case order or binding drifted")
    return document


def validate_tolerance(repo: Path = ROOT) -> dict[str, Any]:
    document = read_json(repo / TOLERANCE_PATH)
    _schema_validate(repo, "tolerance", document)
    if document["approval"]["calibration_candidate_sha256"] is not None:
        raise ContractError("G2 tolerance is candidate-calibrated instead of pre-registered")
    return document


def _u64(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= (1 << 64) - 1:
        raise ContractError(f"{label} is outside u64")
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} contains a symlink component: {current}")


def _stable_file_bytes(path: Path, label: str) -> bytes:
    """Read one regular file and reject replacement/truncation during the read."""

    _reject_symlink_components(path, label)
    try:
        before = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} cannot be stated: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise ContractError(f"{label} changed before it was read")
            data = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ContractError(f"{label} cannot be read: {exc}") from exc
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ContractError(f"{label} was replaced or changed while it was read")
    if len(data) != before.st_size:
        raise ContractError(f"{label} was truncated while it was read")
    return data


def _build_inputs_manifest(repo: Path = ROOT, document: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    if document is None:
        document = read_json(repo / G2_BUILD_INPUTS_PATH)
    if set(document) != {"schema_version", "identity_schema", "role", "binary_name", "source_path", "source_order_sha256", "source_paths"}:
        raise ContractError("G2 build-input manifest has unknown or missing fields")
    if document["schema_version"] != "rmsnorm-g2-build-inputs-v1" or document["identity_schema"] != G2_IDENTITY_SCHEMA:
        raise ContractError("G2 build-input manifest schema identity drifted")
    if document["role"] != G2_ROLE or document["binary_name"] != G2_BINARY or document["source_path"] != G2_SOURCE_PATH:
        raise ContractError("G2 build-input manifest role/name/source binding drifted")
    if not isinstance(document["source_order_sha256"], str) or len(document["source_order_sha256"]) != 64 or document["source_order_sha256"].lower() != document["source_order_sha256"] or any(char not in "0123456789abcdef" for char in document["source_order_sha256"]):
        raise ContractError("G2 build-input source order digest is malformed")
    entries = document["source_paths"]
    if not isinstance(entries, list) or not entries:
        raise ContractError("G2 build-input source_paths is empty or not a list")
    paths: list[str] = []
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"order", "path"} or entry["order"] != order:
            raise ContractError("G2 build-input source order is not canonical")
        path = entry["path"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ContractError("G2 build-input path is not a safe repository-relative path")
        if path in paths:
            raise ContractError("G2 build-input paths are duplicated")
        paths.append(path)
    if paths[0] != G2_SOURCE_PATH or paths[-1] != "native/hip/src/rmsnorm_kernel_internal.hpp":
        raise ContractError("G2 build-input source boundary drifted")
    if sha256_json(paths) != document["source_order_sha256"]:
        raise ContractError("G2 build-input source order digest is stale")
    return tuple(paths)


def _source_set(repo: Path = ROOT) -> dict[str, Any]:
    paths = _build_inputs_manifest(repo)
    files = [{"path": path, "sha256": _sha256(_stable_file_bytes(repo / path, f"G2 source {path}"))} for path in paths]
    return {"canonical_order": list(paths), "files": files, "source_set_sha256": sha256_json(files)}


def expected_build_identity(repo: Path = ROOT) -> dict[str, Any]:
    source_set = _source_set(repo)
    identity = {
        "binary_name": G2_BINARY,
        "identity_schema": G2_IDENTITY_SCHEMA,
        "role": G2_ROLE,
        "source_order_sha256": read_json(repo / G2_BUILD_INPUTS_PATH)["source_order_sha256"],
        "source_path": G2_SOURCE_PATH,
        "source_set_manifest_sha256": sha256_file(repo / G2_BUILD_INPUTS_PATH),
        "source_set_sha256": source_set["source_set_sha256"],
        "source_sha256": source_set["files"][0]["sha256"],
    }
    return {"identity": identity, "identity_sha256": _sha256(canonical_bytes(identity)), "marker": G2_IDENTITY_MARKER}


def query_build_identity(binary: Path, repo: Path = ROOT) -> dict[str, Any]:
    """Require the exact canonical control-plane response from one executable."""

    _require_executable(binary, "G2 identity-query executable")
    binary_bytes = _stable_file_bytes(binary, "G2 identity-query executable")
    _validate_builder_owned_output(binary_bytes, repo)
    query_environment = os.environ.copy()
    query_environment["LD_LIBRARY_PATH"] = G2_RUNTIME_LD_LIBRARY_PATH
    try:
        completed = subprocess.run(
            [str(binary), "--query-build-identity"],
            cwd=repo,
            env=query_environment,
            capture_output=True,
            check=False,
            timeout=5,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"G2 executable identity query failed: {exc}") from exc
    stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout).encode()
    stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr).encode()
    expected = expected_build_identity(repo)["identity"]
    expected_bytes = canonical_bytes(expected)
    if completed.returncode != 0 or stderr != b"" or stdout != expected_bytes:
        raise ContractError(
            "G2 executable identity query must exit 0 with empty stderr and one exact canonical JSON line"
        )
    try:
        identity = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"G2 executable identity query is malformed: {exc}") from exc
    if identity != expected:
        raise ContractError("G2 executable identity query does not match the recomputed source-set identity")
    return identity


def _embedded_build_identity(binary_bytes: bytes) -> dict[str, Any]:
    marker = G2_IDENTITY_MARKER
    positions = []
    offset = 0
    while True:
        found = binary_bytes.find(marker, offset)
        if found < 0:
            break
        positions.append(found)
        offset = found + len(marker)
    if len(positions) != 1:
        raise ContractError("G2 executable has missing or duplicate build identity marker")
    start = positions[0] + len(marker)
    end = binary_bytes.find(b"\n", start)
    if end < 0:
        raise ContractError("G2 executable has a malformed build identity record")
    encoded = binary_bytes[start : end + 1]
    try:
        document = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"G2 executable build identity is malformed: {exc}") from exc
    if not isinstance(document, dict) or canonical_bytes(document) != encoded:
        raise ContractError("G2 executable build identity is not canonical JSON")
    return document


def _validate_embedded_build_identity(binary_bytes: bytes, repo: Path) -> dict[str, Any]:
    expected = expected_build_identity(repo)
    actual = _embedded_build_identity(binary_bytes)
    if actual != expected["identity"]:
        raise ContractError("G2 executable build identity does not match the dedicated role/source/source-set")
    return {**expected, "embedded": actual}


def _nonzero_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value or any(char not in "0123456789abcdef" for char in value) or value == "0" * 64:
        raise ContractError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _canonical_sidecar(binary_digest: str, binary_name: str) -> bytes:
    return f"{binary_digest}  {binary_name}\n".encode("ascii")


def _require_executable(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} cannot be stated: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    if metadata.st_mode & 0o111 == 0:
        raise ContractError(f"{label} is not executable")


def builder_output_path(repo: Path = ROOT) -> Path:
    return repo / G2_BUILDER_OUTPUT_PATH


def g2_build_environment(target: str) -> dict[str, str]:
    if target not in TARGETS:
        raise ContractError("G2 build environment target is not canonical")
    return {
        "ROCM_PATH": "/opt/rocm",
        "HIP_PATH": "/opt/rocm",
        "SLLM_HIP_COMPILER": "/opt/rocm/bin/amdclang++",
        "CMAKE_HIP_ARCHITECTURES": target,
        "SLLM_HIP_CODEGEN_FEATURES": G2_CODEGEN_FEATURES,
        "SLLM_ENABLE_HIP_RUNTIME": "1",
        "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "0",
        "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
    }


def _validate_builder_owned_output(binary_bytes: bytes, repo: Path) -> None:
    """Bind staged evidence to the output discovered from the fixed builder path.

    The marker and sidecar are claims.  The canonical Cargo output is the
    independently observed authority for the trusted single-maintainer host
    contract; same-UID concurrent replacement is intentionally outside A3a.
    """

    reference_path = builder_output_path(repo)
    _require_executable(reference_path, "builder-owned G2 output")
    reference_bytes = _stable_file_bytes(reference_path, "builder-owned G2 output")
    _validate_embedded_build_identity(reference_bytes, repo)
    if binary_bytes != reference_bytes:
        raise ContractError("G2 artifact binary is not the exact output of the fixed Cargo builder")


def _validate_binary_files(
    binary_path: Path,
    document: Mapping[str, Any],
    *,
    repo: Path = ROOT,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    binary = document["binary"]
    if binary_path.name != G2_BINARY:
        raise ContractError("G2 artifact binary path is renamed or is not the dedicated binary")
    sidecar = binary_path.with_name(G2_SIDECAR) if sidecar_path is None else sidecar_path
    if sidecar != binary_path.with_name(G2_SIDECAR) or sidecar.name != G2_SIDECAR:
        raise ContractError("G2 artifact sidecar path is not the canonical sibling")
    _require_executable(binary_path, "G2 dedicated binary")
    binary_bytes = _stable_file_bytes(binary_path, "G2 dedicated binary")
    binary_digest = _sha256(binary_bytes)
    if len(binary_bytes) != binary["size_bytes"] or binary_digest != binary["sha256"]:
        raise ContractError("G2 artifact binary size or SHA-256 does not match the actual file")
    _validate_builder_owned_output(binary_bytes, repo)
    identity = _validate_embedded_build_identity(binary_bytes, repo)
    sidecar_bytes = _stable_file_bytes(sidecar, "G2 dedicated binary sidecar")
    if sidecar_bytes != _canonical_sidecar(binary_digest, G2_BINARY) or _sha256(sidecar_bytes) != binary["sidecar_sha256"]:
        raise ContractError("G2 artifact sidecar is not the canonical digest of the actual binary")
    return identity


def validate_slice_record(record: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    _schema_validate(repo, "slice", record)
    tensor = record["tensor"]
    if tensor["data_buffer_start"] != tensor["header_length_bytes"] + tensor["header_length_field_bytes"]:
        raise ContractError("G2 slice header/data-buffer arithmetic drifted")
    start, end = map(lambda value: _u64(value, "slice offset"), tensor["data_offsets"])
    absolute_start, absolute_end = map(lambda value: _u64(value, "absolute slice offset"), tensor["absolute_byte_range"])
    if not start < end or end - start != BYTE_SIZE or absolute_end - absolute_start != BYTE_SIZE:
        raise ContractError("G2 slice range or byte size is invalid")
    if tensor["data_buffer_start"] + start != absolute_start or tensor["data_buffer_start"] + end != absolute_end:
        raise ContractError("G2 slice relative/absolute offset binding is invalid")
    if tensor["byte_size"] != BYTE_SIZE or tensor["shape"] != [2560] or tensor["dtype"] != "BF16":
        raise ContractError("G2 slice shape/dtype/size is not locked BF16[2560]")
    if record["source"]["model_lock_sha256"] != sha256_file(repo / MODEL_LOCK_PATH) or record["source"]["model_lock_fingerprint"] != MODEL_LOCK_FINGERPRINT or record["source"]["resolved_revision"] != RESOLVED_REVISION:
        raise ContractError("G2 slice model-lock identity drifted")
    recipe = record["recipe"]
    if recipe["synthetic_fixture_only"] is True:
        if recipe["extractor"] != "sllm-g2-synthetic-safetensors-extractor":
            raise ContractError("G2 synthetic slice recipe has an unknown extractor")
    elif recipe["synthetic_fixture_only"] is False:
        if recipe["extractor"] != "sllm-g2-verified-read-only-safetensors-extractor":
            raise ContractError("G2 verified cache slice recipe has an unknown extractor")
        if recipe["arguments"] != ["--cache-root", "<verified-read-only-cache-root>", "--tensor", TENSOR_NAME]:
            raise ContractError("G2 verified cache slice recipe must not record a concrete cache path")
    else:
        raise ContractError("G2 slice recipe synthetic/verified mode is not boolean")
    if record["recipe"]["script_sha256"] != sha256_file(repo / "ci/tools/extract_rmsnorm_g2_slice.py"):
        raise ContractError("G2 slice extractor script hash is stale")
    if record["storage"] != {"raw_slice_stored": False, "raw_slice_uploaded": False, "raw_model_stored": False, "raw_model_uploaded": False, "path_recorded": False}:
        raise ContractError("G2 raw model/slice storage policy was weakened")
    return dict(record)


def _read_fixture(path: Path) -> tuple[dict[str, Any], bytes]:
    if any(part.lower() in {"model", "models", "cache", ".cache", "huggingface", "weights"} for part in path.resolve().parts):
        raise ContractError("G2 synthetic extractor refuses model/cache/weights paths")
    _reject_symlink_components(path, "G2 synthetic fixture")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"cannot stat synthetic G2 fixture: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ContractError("G2 fixture must be a regular non-symlink file")
    if metadata.st_size != ABSOLUTE_RANGE[1]:
        raise ContractError("synthetic G2 fixture has trailing bytes or is truncated")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns):
                raise ContractError("synthetic G2 fixture changed before it was read")
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ContractError("G2 synthetic fixture header length is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length != HEADER_LENGTH:
                raise ContractError("G2 synthetic fixture header length is not 79064")
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise ContractError("G2 synthetic fixture header is truncated")
            stream.seek(ABSOLUTE_RANGE[0])
            payload = stream.read(BYTE_SIZE)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ContractError(f"cannot read synthetic G2 fixture: {exc}") from exc
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns):
        raise ContractError("synthetic G2 fixture was replaced or changed while it was read")
    if len(payload) != BYTE_SIZE:
        raise ContractError("synthetic G2 fixture slice is truncated")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"synthetic G2 fixture header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("__metadata__", {}).get("sllm_fixture") != SYNTHETIC_MARKER:
        raise ContractError("G2 extractor refuses unmarked or real safetensors input")
    if set(header) != {"__metadata__", TENSOR_NAME}:
        raise ContractError("synthetic G2 fixture has an unknown or missing tensor")
    return header, payload


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"safetensors header contains duplicate key: {key}")
        result[key] = value
    return result


def _read_verified_fd_slice(
    fd: int,
    *,
    file_size: int,
    absolute_start: int,
    absolute_end: int,
    expected_sha256: str,
    label: str,
    capture_payload: bool = True,
) -> bytes:
    if absolute_start < 0 or absolute_end <= absolute_start or absolute_end > file_size:
        raise ContractError(f"{label} bounded range is outside the verified file")
    before = os.fstat(fd)
    if before.st_size != file_size or not stat.S_ISREG(before.st_mode):
        raise ContractError(f"{label} changed before same-FD verification")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = file_size
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            raise ContractError(f"{label} was truncated during same-FD verification")
        digest.update(chunk)
        remaining -= len(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ContractError(f"{label} SHA-256 does not match the fixed model lock")
    payload = os.pread(fd, absolute_end - absolute_start, absolute_start) if capture_payload else b""
    after = os.fstat(fd)
    before_tuple = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_tuple = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_tuple != after_tuple or (capture_payload and len(payload) != absolute_end - absolute_start):
        raise ContractError(f"{label} changed while the bounded same-FD slice was read")
    return payload


def _validate_verified_safetensors_header(header_bytes: bytes, *, file_size: int) -> dict[str, Any]:
    if len(header_bytes) < 8:
        raise ContractError("verified safetensors shard header is truncated")
    header_length = struct.unpack("<Q", header_bytes[:8])[0]
    if header_length != HEADER_LENGTH:
        raise ContractError("verified safetensors shard header length is not locked")
    if len(header_bytes) != 8 + header_length:
        raise ContractError("verified safetensors shard header read is incomplete")
    try:
        header = json.loads(header_bytes[8:].decode("utf-8"), object_pairs_hook=_duplicate_rejecting_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"verified safetensors shard header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or ("__metadata__" in header and not isinstance(header["__metadata__"], dict)):
        raise ContractError("verified safetensors shard metadata is malformed")
    if any(not isinstance(value, str) for value in header.get("__metadata__", {}).values()):
        raise ContractError("verified safetensors shard metadata values must be strings")
    data_start = 8 + header_length
    ranges: list[tuple[int, int, str]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ContractError(f"verified safetensors tensor metadata is not closed: {name}")
        offsets = tensor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets):
            raise ContractError(f"verified safetensors tensor offsets are malformed: {name}")
        start, end = offsets
        if not 0 <= start < end or data_start + end > file_size:
            raise ContractError(f"verified safetensors tensor range is outside the shard: {name}")
        ranges.append((start, end, name))
    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start != previous_end:
            raise ContractError(f"verified safetensors tensor ranges overlap or leave an unowned gap: {name}")
        previous_end = end
    if ranges and data_start + previous_end != file_size:
        raise ContractError("verified safetensors shard has trailing or unowned tensor bytes")
    if TENSOR_NAME not in header:
        raise ContractError("verified safetensors shard is missing the locked RMSNorm tensor")
    return header


def extract_verified_slice_payload(cache_root: Path, record: Mapping[str, Any], repo: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    """Verify the complete locked cache, then extract the locked range from one FD.

    This is deliberately not used by the host unit tests with the real model
    cache.  It is the production entry point for a pre-verified, read-only
    cache and never creates a slice file.
    """

    validate_slice_record(record, repo)
    if record["recipe"]["synthetic_fixture_only"] is not False:
        raise ContractError("verified cache extraction requires a non-synthetic slice recipe")
    _reject_symlink_components(cache_root, "G2 verified cache root")
    root_metadata = cache_root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or cache_root.is_symlink() or root_metadata.st_mode & 0o222:
        raise ContractError("G2 verified cache root must be a read-only regular directory")
    lock = read_json(repo / MODEL_LOCK_PATH)
    if not isinstance(lock, dict) or lock.get("schema_version") != "model-lock-v1" or not isinstance(lock.get("model"), dict):
        raise ContractError("G2 model lock is not the fixed model-lock-v1 object")
    lock_files = lock["model"].get("files")
    if not isinstance(lock_files, list) or not lock_files:
        raise ContractError("G2 model lock has no complete file set")
    expected: dict[str, dict[str, Any]] = {}
    for entry in lock_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or entry["path"] in expected:
            raise ContractError("G2 model lock contains duplicate or malformed file entries")
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ContractError("G2 model lock contains an unsafe cache-relative path")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError("G2 model lock contains a malformed file SHA-256")
        lfs_oid = entry.get("lfs_oid")
        if lfs_oid is not None and lfs_oid != f"sha256:{digest}":
            raise ContractError("G2 model lock LFS identity does not match the file SHA-256")
        expected[entry["path"]] = entry
    discovered: set[str] = set()
    for path in cache_root.rglob("*"):
        relative = path.relative_to(cache_root).as_posix()
        if path.is_symlink():
            raise ContractError(f"G2 verified cache contains a symlink: {relative}")
        if path.is_file():
            discovered.add(relative)
        elif path.is_dir():
            if path.stat().st_mode & 0o222:
                raise ContractError(f"G2 verified cache directory is writable: {relative}")
        else:
            raise ContractError(f"G2 verified cache contains a non-regular entry: {relative}")
    if discovered != set(expected):
        raise ContractError("G2 verified cache file set differs from the complete model lock")
    shard_entry = expected.get(SOURCE_SHARD)
    if shard_entry is None:
        raise ContractError("G2 model lock does not contain the locked source shard")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)

    def open_locked_path(relative: str) -> int:
        parts = Path(relative).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ContractError(f"G2 verified cache path is unsafe: {relative}")
        try:
            parent_fd = os.open(cache_root, directory_flags)
        except OSError as exc:
            raise ContractError(f"G2 verified cache root cannot be opened without following symlinks: {exc}") from exc
        try:
            root_opened = os.fstat(parent_fd)
            if (root_opened.st_dev, root_opened.st_ino, root_opened.st_size, root_opened.st_mtime_ns, root_opened.st_ctime_ns) != (root_metadata.st_dev, root_metadata.st_ino, root_metadata.st_size, root_metadata.st_mtime_ns, root_metadata.st_ctime_ns):
                raise ContractError("G2 verified cache root changed before file verification")
            for component in parts[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            return os.open(parts[-1], flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ContractError(f"G2 verified cache path cannot be opened without following symlinks: {relative}: {exc}") from exc
        finally:
            os.close(parent_fd)

    def verify_locked_file(relative: str, entry: Mapping[str, Any]) -> int:
        path = cache_root / relative
        _reject_symlink_components(path, f"G2 verified cache file {relative}")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_mode & 0o222:
            raise ContractError(f"G2 verified cache file is not a read-only regular file: {relative}")
        try:
            local_fd = open_locked_path(relative)
        except ContractError:
            raise
        try:
            opened = os.fstat(local_fd)
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns) or opened.st_size != entry.get("size_bytes"):
                raise ContractError(f"G2 verified cache file changed before hashing: {relative}")
            _read_verified_fd_slice(
                local_fd,
                file_size=opened.st_size,
                absolute_start=0,
                absolute_end=opened.st_size,
                expected_sha256=entry["sha256"],
                label=f"G2 verified cache file {relative}",
                capture_payload=False,
            )
            return local_fd
        except BaseException:
            os.close(local_fd)
            raise

    for relative, entry in expected.items():
        if relative != SOURCE_SHARD:
            fd = verify_locked_file(relative, entry)
            os.close(fd)

    try:
        fd = verify_locked_file(SOURCE_SHARD, shard_entry)
    except OSError as exc:
        raise ContractError(f"G2 verified source shard cannot be opened without following symlinks: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if metadata.st_size != shard_entry.get("size_bytes"):
            raise ContractError("G2 source shard size differs from the fixed model lock")
        payload = _read_verified_fd_slice(
            fd,
            file_size=metadata.st_size,
            absolute_start=ABSOLUTE_RANGE[0],
            absolute_end=ABSOLUTE_RANGE[1],
            expected_sha256=shard_entry.get("sha256", ""),
            label="G2 verified source shard",
        )
        os.lseek(fd, 0, os.SEEK_SET)
        header_bytes = os.read(fd, 8 + HEADER_LENGTH)
        header_after = os.fstat(fd)
        if (header_after.st_dev, header_after.st_ino, header_after.st_size, header_after.st_mtime_ns, header_after.st_ctime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns):
            raise ContractError("G2 source shard changed while its same-FD header was read")
        header = _validate_verified_safetensors_header(header_bytes, file_size=metadata.st_size)
        tensor = header[TENSOR_NAME]
        if tensor != {"dtype": "BF16", "shape": [2560], "data_offsets": list(DATA_OFFSETS)}:
            raise ContractError("verified safetensors tensor metadata differs from the fixed lock")
    finally:
        os.close(fd)
    result = dict(record)
    result["output"] = {"size_bytes": BYTE_SIZE, "sha256": _sha256(payload)}
    return validate_slice_record(result, repo), payload


def extract_synthetic_slice(path: Path, record: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    """Validate and hash a synthetic fixture's locked byte range only.

    No output file is produced.  In particular, this function never accepts a
    cache root and never places raw slice bytes into a report or artifact.
    """

    validate_slice_record(record, repo)
    header, payload = _read_fixture(path)
    tensor = header[TENSOR_NAME]
    if set(tensor) != {"dtype", "shape", "data_offsets"} or tensor != {"dtype": "BF16", "shape": [2560], "data_offsets": list(DATA_OFFSETS)}:
        raise ContractError("synthetic G2 fixture tensor metadata differs from the locked slice")
    if DATA_BUFFER_START != HEADER_LENGTH + 8:
        raise ContractError("synthetic G2 fixture offset arithmetic is out of range")
    result = dict(record)
    result["output"] = {"size_bytes": BYTE_SIZE, "sha256": _sha256(payload)}
    return validate_slice_record(result, repo)


def extract_synthetic_slice_payload(path: Path, record: Mapping[str, Any], repo: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    """Return the validated record plus the exact extractor payload in memory.

    The caller may pass the payload through an anonymous file descriptor for a
    GPU child.  No raw slice file is created or retained by this helper.
    """

    validate_slice_record(record, repo)
    header, payload = _read_fixture(path)
    tensor = header[TENSOR_NAME]
    if set(tensor) != {"dtype", "shape", "data_offsets"} or tensor != {"dtype": "BF16", "shape": [2560], "data_offsets": list(DATA_OFFSETS)}:
        raise ContractError("synthetic G2 fixture tensor metadata differs from the locked slice")
    result = dict(record)
    result["output"] = {"size_bytes": BYTE_SIZE, "sha256": _sha256(payload)}
    return validate_slice_record(result, repo), payload


def _validate_prerequisites(
    prerequisites: list[Mapping[str, Any]],
    *,
    aggregate: bool = False,
    candidate: Mapping[str, Any] | None = None,
    target: str | None = None,
) -> None:
    expected = PREREQUISITE_KINDS
    expected_count = len(expected) * (2 if aggregate else 1)
    if len(prerequisites) != expected_count or [item.get("kind") for item in prerequisites] != list(expected) * (2 if aggregate else 1):
        raise ContractError("G2 prerequisite bindings are missing or duplicated")
    expected_state = "bound-not-executed-by-g2-aggregate" if aggregate else "bound-not-executed-by-g2"
    if any(item.get("state") != expected_state for item in prerequisites):
        raise ContractError("G2 must bind prerequisites without executing them")
    if candidate is not None:
        digest = candidate_sha256(candidate)
        for index, item in enumerate(prerequisites):
            expected_target = TARGETS[index // len(expected)] if aggregate else target
            if item.get("candidate_sha256") != digest:
                raise ContractError("G2 prerequisite candidate identity is stale")
            if aggregate and item.get("target") != expected_target:
                raise ContractError("G2 aggregate prerequisite target order is not canonical")
            item_target = expected_target if expected_target is not None else item.get("target")
            expected_rows = {
                "g0": f"g0-{item_target}",
                "private_g1": f"g1-{item_target}",
                "semantic_g1": f"rmsnorm-semantic-g1-{item_target}",
                "h3": f"h3-rmsnorm-{item_target}",
            }
            if item.get("row_id") != expected_rows[item["kind"]]:
                raise ContractError("G2 prerequisite row ID is not the exact canonical target row")
            _nonzero_sha(item.get("artifact_sha256"), "G2 prerequisite artifact_sha256")
            _nonzero_sha(item.get("report_sha256"), "G2 prerequisite report_sha256")


def validate_artifact(
    document: Mapping[str, Any],
    repo: Path = ROOT,
    *,
    binary_path: Path | None = None,
) -> dict[str, Any]:
    _schema_validate(repo, "artifact", document)
    validate_candidate(document["candidate"], repo)
    if document["row_id"] != f"rmsnorm-g2-{document['target']}":
        raise ContractError("G2 artifact row/target binding drifted")
    binary = document["binary"]
    if binary["path"] != G2_BINARY or binary["sidecar_path"] != G2_SIDECAR or binary["g2_binary_name"] != G2_BINARY:
        raise ContractError("G2 artifact binary identity drifted")
    if (
        binary["build_command"] != list(G2_BUILD_COMMAND)
        or binary["build_profile"] != G2_BUILD_PROFILE
        or binary["builder_output_path"] != G2_BUILDER_OUTPUT_PATH
        or binary["build_environment"] != g2_build_environment(document["target"])
    ):
        raise ContractError("G2 artifact builder provenance is not the fixed offline Cargo invocation")
    if binary["source_path"] != G2_SOURCE_PATH or binary["source_sha256"] == "0" * 64:
        raise ContractError("G2 artifact is not bound to the dedicated Rust source")
    source_set = _source_set(repo)
    if binary["build_source_set"] != source_set or binary["source_sha256"] != source_set["files"][0]["sha256"]:
        raise ContractError("G2 artifact build source-set identity is stale or substituted")
    expected_identity = expected_build_identity(repo)
    expected_artifact_identity = {**expected_identity["identity"], "identity_sha256": expected_identity["identity_sha256"]}
    if binary["build_identity"] != expected_artifact_identity:
        raise ContractError("G2 artifact build identity claim is stale or substituted")
    if document["artifact_id"] != f"rmsnorm-g2-{document['target']}-{binary['sha256']}":
        raise ContractError("G2 artifact ID is not bound to its binary")
    if binary["path"].endswith("sllm-rmsnorm-g1-evidence") or "h3" in binary["path"].lower():
        raise ContractError("G2 artifact substituted a G1/H3 binary")
    if document["scope"] != {"model_used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False}:
        raise ContractError("G2 artifact scope is not the dedicated HIP/no-fallback contract")
    _validate_prerequisites(document["prerequisites"], candidate=document["candidate"], target=document["target"])
    actual_binary = repo / binary["path"] if binary_path is None else binary_path
    embedded_identity = _validate_binary_files(actual_binary, document, repo=repo)
    if embedded_identity["identity"] != expected_identity["identity"] or embedded_identity["identity_sha256"] != expected_identity["identity_sha256"]:
        raise ContractError("G2 executable identity is not bound to the artifact identity")
    return dict(document)


def validate_report(document: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    _schema_validate(repo, "report", document)
    validate_candidate(document["candidate"], repo)
    target = document["target"]
    if document["row_id"] != f"rmsnorm-g2-{target}" or document["device"]["target"] != target:
        raise ContractError("G2 report row/device/target binding drifted")
    expected_device = {**_expected_target(target), "target": target}
    if document["device"] != expected_device:
        raise ContractError("G2 report device identity drifted from the canonical matrix")
    if document["tree_oid"] != document["candidate"]["git_tree_oid"]:
        raise ContractError("G2 report tree identity is not bound to its candidate")
    if document["model"]["used"] is not True or document["model"]["full_model_used"] or document["model"]["tokenizer_used"] or document["model"]["generation_used"]:
        raise ContractError("G2 report model scope is invalid")
    if document["model"]["lock_sha256"] != sha256_file(repo / MODEL_LOCK_PATH) or document["tolerance"]["schema_sha256"] != sha256_file(repo / TOLERANCE_PATH) or document["artifact"]["artifact_schema_sha256"] != sha256_file(repo / SCHEMAS["artifact"]):
        raise ContractError("G2 report contract schema/model hashes are stale")
    for label in ("model.lock_sha256", "tolerance.schema_sha256", "artifact.artifact_schema_sha256", "artifact.binary_sha256", "artifact.binary_sidecar_sha256", "artifact.binary_source_sha256", "artifact.binary_source_set_sha256"):
        section, key = label.split(".")
        _nonzero_sha(document[section][key], f"G2 report {label}")
    source_set = _source_set(repo)
    if document["artifact"]["binary_source_sha256"] != source_set["files"][0]["sha256"] or document["artifact"]["binary_source_set_sha256"] != source_set["source_set_sha256"]:
        raise ContractError("G2 report source identity is stale or substituted")
    _nonzero_sha(document["model"]["slice"]["sha256"], "G2 report model.slice.sha256")
    _nonzero_sha(document["model"]["slice"]["recipe_sha256"], "G2 report model.slice.recipe_sha256")
    if document["health_pre"]["target"] != target or document["health_post"]["target"] != target:
        raise ContractError("G2 report health evidence targets the wrong device")
    if document["scope"]["selected_backend"] != "hip" or not document["scope"]["model_used"] or document["scope"]["full_model_used"] or document["scope"]["tokenizer_used"] or document["scope"]["generation_used"] or document["scope"]["fallback_used"] or document["scope"]["cpu_fallback_used"] or document["scope"]["dispatch_count"] != document["dispatch"]["dispatch_count"] or document["dispatch"]["backend"] != "hip" or document["dispatch"]["kernel_id"] != 1 or document["dispatch"]["kernel_symbol"] != "rmsnorm.baseline.wave32.v1" or document["dispatch"]["device_symbol"] != "sllm_rmsnorm_baseline_wave32_v1" or document["dispatch"]["fallback_used"] or document["dispatch"]["dispatch_count"] == 0 and document["state"] == "PASS":
        raise ContractError("G2 report backend/fallback/model scope is invalid")
    _validate_prerequisites(document["prerequisites"], candidate=document["candidate"], target=document["target"])
    cases = document["cases"]
    if [case["order"] for case in cases] != list(range(6)) or [case["id"] for case in cases] != list(CASE_IDS):
        raise ContractError("G2 report cases are missing, duplicated, stale, or out of order")
    if [case["rows"] for case in cases] != list(CASE_ROWS) or [case["input_seed"] for case in cases] != list(CASE_SEEDS) or any(case["n"] != 2560 or case["classification"] != "finite" for case in cases):
        raise ContractError("G2 report case shape set drifted")
    for case in cases:
        for label in ("request_sha256", "output_sha256", "reference_sha256"):
            if document["state"] == "PASS":
                _nonzero_sha(case[label], f"G2 PASS case {case['id']} {label}")
        for label in ("max_abs_error", "max_rel_error"):
            value = case[label]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ContractError(f"G2 case {case['id']} has a non-finite or negative {label}")
    if document["collection"] != {"expected_cases": 6, "collected_cases": 6, "passed_cases": sum(case["state"] == "PASS" for case in cases), "failed_cases": sum(case["state"] == "FAIL" for case in cases), "expected_rows": 1, "collected_rows": 1}:
        raise ContractError("G2 report collection counts are inconsistent")
    if document["state"] == "PASS":
        if (
            document["execution"]["exit_code"] != 0
            or document["execution"]["timed_out"]
            or document["execution"]["crashed"]
            or document["execution"]["failure_reason"] != ""
            or document["execution"]["protocol_sha256"] == "0" * 64
            or any(document[name]["state"] != "OK" or not document[name]["available"] or not document[name]["reliable"] for name in ("health_pre", "health_post"))
            or any(document[name]["ras_uncorrectable_count"] != 0 for name in ("health_pre", "health_post"))
            or any(document[name]["state"] != "CLEAN" or document[name]["residual_runner_children"] or document[name]["gpu_processes"] for name in ("process_pre", "process_post"))
            or document["scope"]["dispatch_count"] != 6
            or document["dispatch"]["dispatch_count"] != 6
            or document["collection"]["passed_cases"] != 6
            or document["collection"]["failed_cases"] != 0
            or any(case["state"] != "PASS" or case["dispatch_count"] != 1 or case["fallback_used"] or case["nan_count"] != 0 or case["inf_count"] != 0 or case["timeout"] or case["crashed"] for case in cases)
        ):
            raise ContractError("G2 PASS report has invalid execution, health, process, or dispatch evidence")
    return dict(document)


def validate_aggregate(document: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    _schema_validate(repo, "aggregate", document)
    validate_candidate(document["candidate"], repo)
    rows = document["rows"]
    if [row["order"] for row in rows] != [0, 1] or [row["target"] for row in rows] != list(TARGETS) or [row["row_id"] for row in rows] != list(ROWS):
        raise ContractError("G2 aggregate has missing, duplicate, stale, mixed, or reordered rows")
    _validate_prerequisites(document["prerequisites"], aggregate=True, candidate=document["candidate"])
    expected_digest = candidate_sha256(document["candidate"])
    if document["tree_oid"] != document["candidate"]["git_tree_oid"] or any(row["candidate_sha256"] != expected_digest for row in rows):
        raise ContractError("G2 aggregate candidate/tree digest binding is inconsistent")
    if document["aggregate_id"] != f"rmsnorm-g2-aggregate-{document['candidate']['reviewed_sha']}":
        raise ContractError("G2 aggregate ID is not bound to its candidate")
    for label, expected in (
        ("matrix.sha256", sha256_file(repo / MATRIX_PATH)),
        ("model_lock.sha256", sha256_file(repo / MODEL_LOCK_PATH)),
        ("tolerance.sha256", sha256_file(repo / TOLERANCE_PATH)),
    ):
        section, key = label.split(".")
        _nonzero_sha(document[section][key], f"G2 aggregate {label}")
        if document[section][key] != expected:
            raise ContractError(f"G2 aggregate {label} is stale or substituted")
    expected_counts = {
        "expected_rows": 2,
        "selected_rows": 2,
        "collected_rows": sum(row["collected_cases"] == len(CASE_IDS) for row in rows),
        "passed_rows": sum(row["state"] == "PASS" for row in rows),
        "failed_rows": sum(row["state"] == "FAIL" for row in rows),
        "expected_cases": 12,
        "collected_cases": sum(row["collected_cases"] for row in rows),
    }
    if document["counts"] != expected_counts:
        raise ContractError("G2 aggregate collection/count evidence is inconsistent")
    if document["state"] == "PASS":
        if any(row["state"] != "PASS" or row["collected_cases"] != len(CASE_IDS) or row["dispatch_count"] != len(CASE_IDS) or not row["health_ok"] or not row["process_clean"] or row["fallback_used"] for row in rows):
            raise ContractError("G2 aggregate PASS row evidence is incomplete")
    for row in rows:
        for label in ("report_sha256", "artifact_sha256", "candidate_sha256", "tuple_sha256"):
            _nonzero_sha(row[label], f"G2 aggregate {row['row_id']} {label}")
    return dict(document)


def validate_contracts(repo: Path = ROOT) -> None:
    validate_matrix(repo)
    validate_tolerance(repo)
    for name in ("slice", "artifact", "report", "aggregate"):
        _schema(repo, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        validate_contracts(args.repo.resolve())
    except (ContractError, OSError, ValueError) as exc:
        print(f"G2 contracts: FAIL: {exc}", file=sys.stderr)
        return 1
    print("G2 host contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
