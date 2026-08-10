#!/usr/bin/env python3
"""Host-only, fail-closed contracts for the RMSNorm P0 smoke.

This module validates identities and retained measurement documents.  It does
not query a GPU or execute HIP; only the dedicated producer can emit actual
GPU measurements, and all report/aggregate PASS paths remain fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from common import ContractError, ROOT, read_json, sha256_file, sha256_json  # noqa: E402

TARGETS = ("gfx1030", "gfx1201")
ROWS = tuple(f"rmsnorm-p0-{target}" for target in TARGETS)
DISPATCH_BLOCK_SIZE = 256
WARMUP_ITERATIONS = 5
MEASUREMENT_ITERATIONS = 21
TOTAL_DISPATCHES = 5 * (WARMUP_ITERATIONS + MEASUREMENT_ITERATIONS)
MAX_LATENCY_NS = 15 * 60 * 1_000_000_000
MODEL_LOCK_PATH = "docs/models/locks/qwen3.5-4b-bf16.json"
MODEL_LOCK_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
RESOLVED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MATRIX_PATH = "ci/matrix/rmsnorm-p0-v1.json"
REVIEW_POLICY_PATH = "ci/matrix/rmsnorm-p0-review-policy-v1.json"
P0_BINARY = "sllm-rmsnorm-p0-evidence"
P0_SIDECAR = P0_BINARY + ".sha256"
P0_BINARY_ROLE = "dedicated-p0-public-rmsnorm-producer"
P0_BUILD_COMMAND = (
    "cargo", "+1.97.1", "build", "--locked", "--offline", "--release",
    "--package", "sllm-hip", "--bin", P0_BINARY,
)
P0_CODEGEN_FEATURES = "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0"
P0_RUNTIME_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib"
P0_BUILD_TIMEOUT_SECONDS = 900.0
P0_BUILD_OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
P0_BUILD_KILL_GRACE_SECONDS = 2.0
P0_BUILD_LIMITS = {
    "timeout_seconds": 900,
    "combined_output_bytes": P0_BUILD_OUTPUT_LIMIT_BYTES,
    "start_new_session": True,
    "process_group_cleanup": "term-kill-group-disappearance-v1",
    "termination_grace_seconds": 2,
}
PRODUCER_STATUS = "a5-enabled"
DTYPE_CONTRACT = {
    "activation": "BF16", "weight": "BF16", "output": "BF16",
    "accumulation": "F32", "scale_mode": "offset-one", "epsilon": "1e-6",
}
CASE_SPECS = (
    ("p0-r3-n37", 3, 37, 9301, "synthetic-nonaligned"),
    ("p0-r1-n2560", 1, 2560, 9302, "locked-model-hidden-size"),
    ("p0-r1-n255", 1, 255, 9303, "dispatch-b-minus-1"),
    ("p0-r1-n256", 1, 256, 9304, "dispatch-b"),
    ("p0-r1-n257", 1, 257, 9305, "dispatch-b-plus-1"),
)
CASE_IDS = tuple(item[0] for item in CASE_SPECS)
PREREQUISITE_KINDS = ("g0", "private_g1", "semantic_g1", "g2", "h3")
PUBLIC_PATH = "SemanticOpDescriptor->Backend->sllm-hip->public-C-ABI->native-HIP-registry->rmsnorm.baseline.wave32.v1"


def p0_build_environment(target: str) -> dict[str, str]:
    if target not in TARGETS:
        raise ContractError("P0 build environment target is not canonical")
    return {
        "ROCM_PATH": "/opt/rocm",
        "HIP_PATH": "/opt/rocm",
        "SLLM_HIP_COMPILER": "/opt/rocm/bin/amdclang++",
        "CMAKE_HIP_ARCHITECTURES": target,
        "SLLM_HIP_CODEGEN_FEATURES": P0_CODEGEN_FEATURES,
        "SLLM_ENABLE_HIP_RUNTIME": "0",
        "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "1",
        "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
    }
P0_PUBLIC_PATH_INPUTS_PATH = "ci/matrix/rmsnorm-p0-public-path-inputs-v1.json"
P0_SOURCE_SET_IDENTITY = "rmsnorm-p0-enabled-public-path-source-set-v1"
A5_ENABLEMENT_REQUIREMENTS: tuple[str, ...] = ()
EXPECTED_SOURCE_PATHS = (
    P0_PUBLIC_PATH_INPUTS_PATH,
    "ci/tools/validate_rmsnorm_p0_contracts.py",
    "ci/tools/run_rmsnorm_p0_runtime.py",
    "ci/tools/aggregate_rmsnorm_p0_results.py",
    "ci/tools/build_rmsnorm_p0_runtime.py",
    "Cargo.lock",
    "Cargo.toml",
    "crates/sllm-core/Cargo.toml",
    "crates/sllm-core/src/backend.rs",
    "crates/sllm-core/src/dtype.rs",
    "crates/sllm-core/src/execution.rs",
    "crates/sllm-core/src/fake.rs",
    "crates/sllm-core/src/handles.rs",
    "crates/sllm-core/src/kv_state.rs",
    "crates/sllm-core/src/lib.rs",
    "crates/sllm-core/src/model.rs",
    "crates/sllm-core/src/op.rs",
    "crates/sllm-core/src/registry.rs",
    "crates/sllm-core/src/tensor.rs",
    "crates/sllm-core/src/weights.rs",
    "crates/sllm-hip-sys/Cargo.toml",
    "crates/sllm-hip-sys/build.rs",
    "crates/sllm-hip-sys/src/bindings.rs",
    "crates/sllm-hip-sys/src/evidence_bindings.rs",
    "crates/sllm-hip-sys/src/lib.rs",
    "crates/sllm-hip/Cargo.toml",
    "crates/sllm-hip/src/attention_preprocess.rs",
    "crates/sllm-hip/src/bridge.rs",
    "crates/sllm-hip/src/embedding.rs",
    "crates/sllm-hip/src/elementwise.rs",
    "crates/sllm-hip/src/lib.rs",
    "crates/sllm-hip/src/kv_state.rs",
    "crates/sllm-hip/src/matmul.rs",
    "crates/sllm-hip/src/rmsnorm.rs",
    "crates/sllm-hip/src/runtime.rs",
    "crates/sllm-hip/src/bin/sllm-rmsnorm-p0-evidence.rs",
    "include/sllm/hip.h",
    "include/sllm/sllm.h",
    "native/hip/CMakeLists.txt",
    "native/hip/src/abi_layout_probe.cpp",
    "native/hip/src/attention_preprocess_api.cpp",
    "native/hip/src/attention_preprocess_api.hpp",
    "native/hip/src/attention_preprocess_kernel.hip.cpp",
    "native/hip/src/attention_preprocess_kernel_internal.hpp",
    "native/hip/src/attention_preprocess_runtime.inc",
    "native/hip/src/causal_attention_api.cpp",
    "native/hip/src/causal_attention_api.hpp",
    "native/hip/src/causal_attention_kernel.hip.cpp",
    "native/hip/src/causal_attention_kernel_internal.hpp",
    "native/hip/src/causal_attention_runtime.inc",
    "native/hip/src/embedding_api.cpp",
    "native/hip/src/embedding_api.hpp",
    "native/hip/src/embedding_kernel.hip.cpp",
    "native/hip/src/embedding_kernel_internal.hpp",
    "native/hip/src/embedding_runtime.inc",
    "native/hip/src/elementwise_api.cpp",
    "native/hip/src/elementwise_api.hpp",
    "native/hip/src/elementwise_kernel.hip.cpp",
    "native/hip/src/elementwise_kernel_internal.hpp",
    "native/hip/src/evidence_abi.h",
    "native/hip/src/header_c_compile.c",
    "native/hip/src/header_cpp_compile.cpp",
    "native/hip/src/hip_compile_probe.hip.cpp",
    "native/hip/src/hip_evidence_runtime.hip.cpp",
    "native/hip/src/hip_evidence_stub.cpp",
    "native/hip/src/hip_stub.cpp",
    "native/hip/src/kv_state_api.cpp",
    "native/hip/src/kv_state_api.hpp",
    "native/hip/src/kv_state_kernel.hip.cpp",
    "native/hip/src/kv_state_kernel_internal.hpp",
    "native/hip/src/matmul_api.cpp",
    "native/hip/src/matmul_api.hpp",
    "native/hip/src/matmul_kernel.hip.cpp",
    "native/hip/src/matmul_kernel_internal.hpp",
    "native/hip/src/matmul_runtime.inc",
    "native/hip/src/public_runtime.hip.cpp",
    "native/hip/src/public_runtime_internal.hpp",
    "native/hip/src/public_runtime_stub.cpp",
    "native/hip/src/rmsnorm_api.cpp",
    "native/hip/src/rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_kernel.hip.cpp",
    "native/hip/src/rmsnorm_kernel_internal.hpp",
)
SCHEMAS = {
    "matrix": "ci/schema/rmsnorm-p0-matrix-v1.schema.json",
    "review_policy": "ci/schema/rmsnorm-p0-review-policy-v1.schema.json",
    "artifact": "ci/schema/rmsnorm-p0-artifact-v1.schema.json",
    "runtime_result": "ci/schema/rmsnorm-p0-runtime-result-v1.schema.json",
    "report": "ci/schema/rmsnorm-p0-report-v1.schema.json",
    "disposition": "ci/schema/rmsnorm-p0-review-disposition-v1.schema.json",
    "aggregate": "ci/schema/rmsnorm-p0-aggregate-v1.schema.json",
}


def _schema(repo: Path, name: str) -> dict[str, Any]:
    value = read_json(repo / SCHEMAS[name])
    if not isinstance(value, dict):
        raise ContractError(f"P0 {name} schema is not an object")
    Draft202012Validator.check_schema(value)
    return value


def _schema_validate(repo: Path, name: str, document: Any) -> None:
    errors = sorted(
        Draft202012Validator(
            _schema(repo, name), format_checker=FormatChecker()
        ).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        detail = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:5]
        )
        raise ContractError(f"P0 {name} schema rejected document: {detail}")


def _nonzero_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 64
    ):
        raise ContractError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def candidate_sha256(candidate: Mapping[str, Any]) -> str:
    return sha256_json(dict(candidate))


def validate_candidate(
    candidate: Mapping[str, Any], repo: Path = ROOT, *, strict_git: bool = False
) -> dict[str, Any]:
    required = {
        "reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid",
        "worktree_clean", "revision_input",
    }
    if set(candidate) != required:
        raise ContractError("P0 candidate has missing or unknown fields")
    commits: list[str] = []
    for name in ("reviewed_sha", "tested_sha", "workflow_sha"):
        value = candidate[name]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or value.lower() != value
            or any(char not in "0123456789abcdef" for char in value)
            or value == "0" * 40
        ):
            raise ContractError(f"P0 {name} is not a nonzero full lowercase SHA")
        commits.append(value)
    tree = candidate["git_tree_oid"]
    if (
        not isinstance(tree, str)
        or len(tree) != 40
        or tree.lower() != tree
        or any(char not in "0123456789abcdef" for char in tree)
        or tree == "0" * 40
    ):
        raise ContractError("P0 tree OID is not a nonzero full lowercase identity")
    if len(set(commits)) != 1:
        raise ContractError("P0 reviewed/tested/workflow SHA identities differ")
    if candidate["worktree_clean"] is not True or candidate["revision_input"] != "full-sha":
        raise ContractError("P0 candidate is not a clean full-SHA identity")
    if strict_git:
        def git(*args: str) -> str:
            completed = subprocess.run(
                ["git", *args], cwd=repo, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                raise ContractError(f"cannot verify P0 Git identity: {completed.stderr.strip()}")
            return completed.stdout.strip()

        actual_commit = git("rev-parse", "--verify", "HEAD^{commit}")
        actual_tree = git("rev-parse", "--verify", f"{actual_commit}^{{tree}}")
        if actual_commit != commits[0] or actual_tree != tree:
            raise ContractError("P0 candidate does not match the checkout commit/tree")
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            raise ContractError("P0 evidence requires a clean worktree")
    return dict(candidate)


def _expected_device(target: str) -> dict[str, Any]:
    if target == "gfx1030":
        return {"bdf": "0000:03:00.0", "uuid": "GPU-76a08c022586fed6", "product": "AMD Radeon Pro V620", "physical_hip_index": 1, "logical_device_index": 0}
    if target == "gfx1201":
        return {"bdf": "0000:07:00.0", "uuid": "GPU-a8e9ddefa2d60f55", "product": "AMD Radeon AI PRO R9700", "physical_hip_index": 2, "logical_device_index": 0}
    raise ContractError(f"unsupported P0 target: {target}")


def expected_matrix() -> dict[str, Any]:
    cases = [
        {"order": order, "id": case_id, "rows": rows, "n": n, "input_seed": seed, "classification": classification}
        for order, (case_id, rows, n, seed, classification) in enumerate(CASE_SPECS)
    ]
    return {
        "schema_version": "rmsnorm-p0-matrix-v1", "matrix_id": "rmsnorm-p0-v1",
        "revision": 1, "suite_id": "p0-rmsnorm-runtime-smoke", "tier": "tier_p0",
        "required": True, "serial": True, "dispatch_block_size": DISPATCH_BLOCK_SIZE,
        "timing": {"timing_contract": "rmsnorm-p0-timing-v1", "unit": "ns", "kernel_source": "hip-event-elapsed-time", "wall_source": "steady-clock-monotonic", "warmup_iterations": WARMUP_ITERATIONS, "measurement_iterations": MEASUREMENT_ITERATIONS, "location": "median", "robust_spread": "median-absolute-deviation"},
        "dtype": dict(DTYPE_CONTRACT),
        "scope": {"selected_backend": "hip", "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "gpu_execution": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "optimized_claim": False, "other_engine_comparison": False, "performance_hard_gate": False},
        "cases": cases,
        "targets": [
            {"order": order, "row_id": f"rmsnorm-p0-{target}", "target": target, "device": _expected_device(target), "backend": "hip", "cases": list(CASE_IDS)}
            for order, target in enumerate(TARGETS)
        ],
        "model_lock": {"path": MODEL_LOCK_PATH, "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION},
        "review_policy": REVIEW_POLICY_PATH,
        "runtime_producer": {"binary_name": P0_BINARY, "role": P0_BINARY_ROLE, "result_schema": "rmsnorm-p0-runtime-result-v1", "implementation_status": PRODUCER_STATUS},
        "prerequisites": list(PREREQUISITE_KINDS),
    }


def expected_review_policy() -> dict[str, Any]:
    return {
        "schema_version": "rmsnorm-p0-review-policy-v1", "policy_id": "rmsnorm-p0-review-policy-v1", "revision": 1,
        "performance_sanity_disposition": "review_required",
        "threshold": {"approved": False, "threshold_id": None, "metric_thresholds": []},
        "required_measurement": {"targets": list(TARGETS), "case_ids": list(CASE_IDS), "boundary_triplet": list(CASE_IDS[2:]), "warmup_iterations": WARMUP_ITERATIONS, "measurement_iterations": MEASUREMENT_ITERATIONS, "metrics": ["kernel_latency_ns", "wall_latency_ns"], "summaries": ["median_ns", "median_absolute_deviation_ns"]},
        "required_review": {"decision": "accept_observation_without_threshold", "fields": ["reviewer", "reason", "reviewed_at", "canonical_rows"]},
        "claims": {"optimized": False, "faster_than_other_engine": False, "performance_hard_gate_established": False},
    }


def validate_matrix(repo: Path = ROOT) -> dict[str, Any]:
    document = read_json(repo / MATRIX_PATH)
    _schema_validate(repo, "matrix", document)
    if document != expected_matrix():
        raise ContractError("P0 matrix target/case/order/timing binding drifted")
    if [item[2] for item in CASE_SPECS[2:]] != [DISPATCH_BLOCK_SIZE - 1, DISPATCH_BLOCK_SIZE, DISPATCH_BLOCK_SIZE + 1]:
        raise ContractError("P0 dispatch boundary is not canonical B-1/B/B+1")
    return document


def validate_review_policy(repo: Path = ROOT) -> dict[str, Any]:
    document = read_json(repo / REVIEW_POLICY_PATH)
    _schema_validate(repo, "review_policy", document)
    if document != expected_review_policy():
        raise ContractError("P0 review-required policy drifted")
    return document


def case_set_sha256(repo: Path = ROOT) -> str:
    return sha256_json(validate_matrix(repo)["cases"])


def public_path_source_paths(
    repo: Path = ROOT, document: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    if document is None:
        document = read_json(repo / P0_PUBLIC_PATH_INPUTS_PATH)
    required = {
        "schema_version", "identity", "public_path", "producer_status",
        "dedicated_producer_included", "a5_enablement_requires",
        "source_order_sha256", "source_paths",
    }
    if set(document) != required:
        raise ContractError("P0 public-path input manifest has unknown or missing fields")
    if (
        document["schema_version"] != "rmsnorm-p0-public-path-inputs-v1"
        or document["identity"] != "rmsnorm-p0-enabled-public-path-v1"
        or document["public_path"] != PUBLIC_PATH
    ):
        raise ContractError("P0 public-path input manifest identity drifted")
    if (
        document["producer_status"] != PRODUCER_STATUS
        or document["dedicated_producer_included"] is not True
        or document["a5_enablement_requires"] != list(A5_ENABLEMENT_REQUIREMENTS)
    ):
        raise ContractError(
            "P0 public-path input manifest does not bind the enabled dedicated producer"
        )
    order_digest = document["source_order_sha256"]
    if (
        not isinstance(order_digest, str)
        or len(order_digest) != 64
        or order_digest.lower() != order_digest
        or any(char not in "0123456789abcdef" for char in order_digest)
    ):
        raise ContractError("P0 public-path source-order digest is malformed")
    entries = document["source_paths"]
    if not isinstance(entries, list) or not entries:
        raise ContractError("P0 public-path source list is empty or not a list")
    paths: list[str] = []
    for order, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"order", "path"}
            or entry["order"] != order
        ):
            raise ContractError("P0 public-path source order is not canonical")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ContractError("P0 public-path source is not a safe repository-relative path")
        if relative in paths:
            raise ContractError("P0 public-path sources are duplicated")
        paths.append(relative)
    if tuple(paths) != EXPECTED_SOURCE_PATHS:
        raise ContractError("P0 public-path source closure is omitted, reordered, or path-mutated")
    if sha256_json(paths) != order_digest:
        raise ContractError("P0 public-path source-order digest is stale")
    return tuple(paths)


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} contains a symlink component: {current}")


def _stable_file_bytes(path: Path, label: str) -> bytes:
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
            before_identity = (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            )
            opened_identity = (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
            )
            if opened_identity != before_identity:
                raise ContractError(f"{label} changed before it was read")
            data = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ContractError(f"{label} cannot be read: {exc}") from exc
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if after_identity != before_identity or len(data) != before.st_size:
        raise ContractError(f"{label} was replaced, changed, or truncated while it was read")
    return data


def source_set(repo: Path = ROOT) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for order, relative in enumerate(public_path_source_paths(repo)):
        source_bytes = _stable_file_bytes(repo / relative, f"P0 source {relative}")
        files.append({
            "order": order,
            "path": relative,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        })
    return {
        "identity": P0_SOURCE_SET_IDENTITY,
        "files": files,
        "sha256": sha256_json(files),
    }


def _canonical_sidecar(binary_sha: str) -> bytes:
    return f"{binary_sha}  {P0_BINARY}\n".encode("ascii")


def _require_regular(path: Path, label: str, *, executable: bool = False) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} cannot be stated: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    if executable and metadata.st_mode & 0o111 == 0:
        raise ContractError(f"{label} must be executable")
    return metadata


def _validate_prerequisites(
    values: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], target: str
) -> None:
    if len(values) != 5 or [value.get("kind") for value in values] != list(PREREQUISITE_KINDS):
        raise ContractError("P0 prerequisites are missing, duplicated, or reordered")
    digest = candidate_sha256(candidate)
    expected_rows = {
        "g0": f"g0-{target}", "private_g1": f"g1-{target}",
        "semantic_g1": f"rmsnorm-semantic-g1-{target}",
        "g2": f"rmsnorm-g2-{target}", "h3": f"h3-rmsnorm-{target}",
    }
    for value in values:
        if value.get("state") != "bound-not-executed-by-p0" or value.get("row_id") != expected_rows[value["kind"]] or value.get("candidate_sha256") != digest:
            raise ContractError("P0 prerequisite candidate/row/state binding is stale")
        _nonzero_sha(value.get("artifact_sha256"), "P0 prerequisite artifact")
        _nonzero_sha(value.get("report_sha256"), "P0 prerequisite report")


def validate_artifact(
    document: Mapping[str, Any], repo: Path = ROOT, *, binary_path: Path | None = None
) -> dict[str, Any]:
    _schema_validate(repo, "artifact", document)
    candidate = validate_candidate(document["candidate"], repo)
    target = document["target"]
    if document["row_id"] != f"rmsnorm-p0-{target}":
        raise ContractError("P0 artifact row/target binding drifted")
    expected_source_set = source_set(repo)
    if document["source_set"] != expected_source_set:
        raise ContractError("P0 artifact public-path/source identity is stale")
    binary = document["binary"]
    actual_binary = binary_path if binary_path is not None else repo / binary["path"]
    metadata = _require_regular(actual_binary, "P0 producer", executable=True)
    if actual_binary.name != P0_BINARY or binary["path"] != P0_BINARY or binary["sidecar_path"] != P0_SIDECAR:
        raise ContractError("P0 artifact substituted a noncanonical producer path")
    actual_sha = sha256_file(actual_binary)
    sidecar = actual_binary.with_name(P0_SIDECAR)
    _require_regular(sidecar, "P0 producer sidecar")
    sidecar_bytes = sidecar.read_bytes()
    if sidecar_bytes != _canonical_sidecar(actual_sha):
        raise ContractError("P0 producer sidecar is noncanonical or stale")
    if binary != {"role": P0_BINARY_ROLE, "path": P0_BINARY, "sidecar_path": P0_SIDECAR, "size_bytes": metadata.st_size, "sha256": actual_sha, "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest()}:
        raise ContractError("P0 artifact binary identity does not match actual files")
    expected_build = {
        "builder": "ci/tools/build_rmsnorm_p0_runtime.py",
        "command": list(P0_BUILD_COMMAND),
        "profile": "release",
        "binary_name": P0_BINARY,
        "output_path": P0_BINARY,
        "fresh_output": True,
        "substitution_rejected": True,
        "environment": p0_build_environment(target),
        "limits": P0_BUILD_LIMITS,
    }
    if document["build"] != expected_build:
        raise ContractError("P0 artifact build identity is missing or substituted")
    expected_contract = {"public_path": PUBLIC_PATH, "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": DISPATCH_BLOCK_SIZE, "timing_contract": "rmsnorm-p0-timing-v1", "dtype": DTYPE_CONTRACT, "producer_status": PRODUCER_STATUS}
    if document["execution_contract"] != expected_contract:
        raise ContractError("P0 artifact execution/timing contract drifted")
    expected_scope = {"selected_backend": "hip", "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False}
    if document["scope"] != expected_scope:
        raise ContractError("P0 artifact scope is not HIP public RMSNorm without fallback")
    if document["artifact_id"] != f"rmsnorm-p0-{target}-{actual_sha}":
        raise ContractError("P0 artifact ID is not bound to its binary")
    _validate_prerequisites(document["prerequisites"], candidate, target)
    return dict(document)


def artifact_summary(
    artifact: Mapping[str, Any], artifact_sha: str
) -> dict[str, Any]:
    return {
        "artifact_id": artifact["artifact_id"], "artifact_sha256": artifact_sha,
        "binary_sha256": artifact["binary"]["sha256"],
        "binary_sidecar_sha256": artifact["binary"]["sidecar_sha256"],
        "source_set_sha256": artifact["source_set"]["sha256"],
        "binary_role": P0_BINARY_ROLE,
    }


def _strict_positive_ns(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_LATENCY_NS:
        raise ContractError(f"{label} is nonfinite, nonpositive, noninteger, or out of range")
    return value


def _median(values: Sequence[int]) -> int:
    result = median(values)
    if not isinstance(result, int):
        raise ContractError("P0 odd-sized median did not produce an integer")
    return result


def _mad(values: Sequence[int]) -> int:
    center = _median(values)
    return _median([abs(value - center) for value in values])


def _validate_measurements(cases: Sequence[Mapping[str, Any]]) -> None:
    if len(cases) != len(CASE_SPECS):
        raise ContractError("P0 measurement case collection is incomplete")
    dispatch_ids: list[int] = []
    for order, (case, spec) in enumerate(zip(cases, CASE_SPECS)):
        case_id, rows, n, seed, classification = spec
        expected = {"order": order, "id": case_id, "rows": rows, "n": n, "input_seed": seed, "classification": classification}
        if any(case.get(key) != value for key, value in expected.items()) or case.get("state") != "PASS":
            raise ContractError("P0 case order/tuple/classification drifted")
        warmups = case["warmup_dispatches"]
        samples = case["samples"]
        if len(warmups) != WARMUP_ITERATIONS or len(samples) != MEASUREMENT_ITERATIONS:
            raise ContractError("P0 warmup or measurement iteration count drifted")
        for iteration, dispatch in enumerate(warmups):
            if dispatch["iteration"] != iteration:
                raise ContractError("P0 warmup iteration order drifted")
            dispatch_ids.append(dispatch["dispatch_id"])
        kernel_values: list[int] = []
        wall_values: list[int] = []
        for iteration, sample in enumerate(samples):
            if sample["iteration"] != iteration:
                raise ContractError("P0 measurement iteration order drifted")
            kernel = _strict_positive_ns(sample["kernel_latency_ns"], "P0 kernel latency")
            wall = _strict_positive_ns(sample["wall_latency_ns"], "P0 wall latency")
            if wall < kernel:
                raise ContractError("P0 wall latency is smaller than kernel latency")
            kernel_values.append(kernel)
            wall_values.append(wall)
            dispatch_ids.append(sample["dispatch_id"])
        expected_summary = {
            "kernel_median_ns": _median(kernel_values),
            "kernel_mad_ns": _mad(kernel_values),
            "wall_median_ns": _median(wall_values),
            "wall_mad_ns": _mad(wall_values),
            "sample_set_sha256": sha256_json(samples),
        }
        if case["summary"] != expected_summary:
            raise ContractError("P0 median/MAD/sample identity is declarative or stale")
    if dispatch_ids != list(range(1, TOTAL_DISPATCHES + 1)):
        raise ContractError("P0 dispatch IDs are zero, duplicate, missing, or nonmonotonic")


def validate_runtime_result(
    document: Mapping[str, Any], artifact: Mapping[str, Any], artifact_sha: str,
    repo: Path = ROOT,
) -> dict[str, Any]:
    _schema_validate(repo, "runtime_result", document)
    candidate = validate_candidate(document["candidate"], repo)
    target = document["target"]
    if document["row_id"] != f"rmsnorm-p0-{target}" or artifact["target"] != target or artifact["candidate"] != candidate:
        raise ContractError("P0 runtime result artifact/candidate/row binding drifted")
    if document["artifact"] != artifact_summary(artifact, artifact_sha):
        raise ContractError("P0 runtime result artifact identity is stale or substituted")
    if document["matrix"] != {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)} or document["case_set_sha256"] != case_set_sha256(repo):
        raise ContractError("P0 runtime result matrix/case-set identity drifted")
    expected_model = {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION, "used": False}
    if document["model_lock"] != expected_model or document["source_set_sha256"] != source_set(repo)["sha256"]:
        raise ContractError("P0 runtime result model-lock/source identity drifted")
    if document["dtype"] != DTYPE_CONTRACT:
        raise ContractError("P0 runtime result dtype/accumulation contract drifted")
    expected_scope = {"selected_backend": "hip", "gpu_execution": True, "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False}
    if document["scope"] != expected_scope:
        raise ContractError("P0 runtime result is non-GPU, fallback, or not the public RMSNorm path")
    if document["device"] != {**_expected_device(target), "target": target}:
        raise ContractError("P0 runtime result device tuple is not canonical")
    if document["timing"] != expected_matrix()["timing"]:
        raise ContractError("P0 runtime result timing contract drifted")
    expected_dispatch = {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": DISPATCH_BLOCK_SIZE, "dispatch_count": TOTAL_DISPATCHES, "fallback_allowed": False, "fallback_used": False}
    if document["dispatch"] != expected_dispatch:
        raise ContractError("P0 runtime result dispatch/kernel/fallback identity drifted")
    _validate_measurements(document["cases"])
    if document["measurement_sha256"] != sha256_json(document["cases"]):
        raise ContractError("P0 runtime result measurement identity is stale")
    _nonzero_sha(document["measurement_sha256"], "P0 runtime measurement")
    return dict(document)


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"{label} is not a strict UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is invalid") from exc
    return parsed.astimezone(timezone.utc)


def _validate_report_identities(document: Mapping[str, Any], repo: Path) -> None:
    if document["matrix"] != {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)}:
        raise ContractError("P0 report matrix identity drifted")
    if document["review_policy"] != {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)}:
        raise ContractError("P0 report review policy identity drifted")
    if document["case_set_sha256"] != case_set_sha256(repo) or document["source_set_sha256"] != source_set(repo)["sha256"]:
        raise ContractError("P0 report case/source identity drifted")
    expected_model = {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION, "used": False}
    if document["model_lock"] != expected_model:
        raise ContractError("P0 report model-lock identity drifted")


def validate_report(document: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    _schema_validate(repo, "report", document)
    candidate = validate_candidate(document["candidate"], repo)
    target = document["target"]
    if document["row_id"] != f"rmsnorm-p0-{target}" or document["tree_oid"] != candidate["git_tree_oid"]:
        raise ContractError("P0 report row/tree/candidate binding drifted")
    if document["device"] != {**_expected_device(target), "target": target}:
        raise ContractError("P0 report device tuple drifted")
    _validate_report_identities(document, repo)
    _validate_prerequisites(document["prerequisites"], candidate, target)
    _nonzero_sha(document["artifact"]["artifact_sha256"], "P0 report artifact")
    if document["artifact"]["source_set_sha256"] != source_set(repo)["sha256"] or document["artifact"]["producer_status"] != PRODUCER_STATUS:
        raise ContractError("P0 report artifact/source producer identity drifted")
    if document["dtype"] != DTYPE_CONTRACT:
        raise ContractError("P0 report dtype/accumulation contract drifted")
    measurements = document["measurements"]
    if measurements:
        _validate_measurements(measurements)
        if document["measurement_sha256"] != sha256_json(measurements):
            raise ContractError("P0 report measurement identity drifted")
        _nonzero_sha(document["runtime_result_sha256"], "P0 report runtime result")
    elif document["measurement_sha256"] != "0" * 64 or document["runtime_result_sha256"] != "0" * 64:
        raise ContractError("P0 empty report carries forged measurement identity")
    collected = len(measurements)
    expected_collection = {"expected_cases": 5, "collected_cases": collected, "passed_cases": collected, "failed_cases": 0 if collected else 5, "warmup_dispatches": collected * WARMUP_ITERATIONS, "measurement_dispatches": collected * MEASUREMENT_ITERATIONS}
    if document["collection"] != expected_collection:
        raise ContractError("P0 report collection counts are inconsistent")
    expected_dispatch_count = TOTAL_DISPATCHES if collected == 5 else 0
    expected_scope = {"selected_backend": "hip", "gpu_execution": collected == 5, "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "optimized_claim": False, "other_engine_comparison": False, "performance_hard_gate": False}
    if document["scope"] != expected_scope:
        raise ContractError("P0 report scope is non-GPU, fallback, or makes an unsupported claim")
    expected_dispatch = {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": DISPATCH_BLOCK_SIZE, "dispatch_count": expected_dispatch_count, "fallback_allowed": False, "fallback_used": False}
    if document["dispatch"] != expected_dispatch:
        raise ContractError("P0 report dispatch/kernel identity is inconsistent")
    started = _parse_time(document["execution"]["started_at"], "P0 started_at")
    finished = _parse_time(document["execution"]["finished_at"], "P0 finished_at")
    if finished < started:
        raise ContractError("P0 report timestamps are reversed")
    for name in ("health_pre", "health_post"):
        if document[name]["target"] != target:
            raise ContractError("P0 health evidence targets another row")
    if measurements and (
        document["execution"]["duration_ns"] <= 0
        or document["execution"]["exit_code"] != 0
        or document["execution"]["timed_out"]
        or document["execution"]["crashed"]
        or document["execution"]["stderr_sha256"] != hashlib.sha256(b"").hexdigest()
        or any(document[name]["state"] != "OK" or not document[name]["available"] or not document[name]["reliable"] for name in ("health_pre", "health_post"))
        or document["health_post"]["ras_uncorrectable_count"] > document["health_pre"]["ras_uncorrectable_count"]
        or any(document[name]["state"] != "CLEAN" or document[name]["residual_runner_children"] or document[name]["gpu_processes"] for name in ("process_pre", "process_post"))
    ):
        raise ContractError("P0 retained measurements lack clean execution/health/process evidence")
    if document["state"] == "PASS":
        if (
            collected != 5
            or document["scope"]["gpu_execution"] is not True
            or document["scope"]["fallback_used"]
            or document["dispatch"]["fallback_used"]
            or document["execution"]["exit_code"] != 0
            or document["execution"]["timed_out"]
            or document["execution"]["crashed"]
            or document["execution"]["stderr_sha256"] != hashlib.sha256(b"").hexdigest()
            or any(document[name]["state"] != "OK" or not document[name]["available"] or not document[name]["reliable"] for name in ("health_pre", "health_post"))
            or document["health_post"]["ras_uncorrectable_count"] > document["health_pre"]["ras_uncorrectable_count"]
            or any(document[name]["state"] != "CLEAN" or document[name]["residual_runner_children"] or document[name]["gpu_processes"] for name in ("process_pre", "process_post"))
        ):
            raise ContractError("P0 PASS report has incomplete GPU/timing/health/process evidence")
    return dict(document)


def _case_summaries(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"order": case["order"], "id": case["id"], **case["summary"]}
        for case in report["measurements"]
    ]


def validate_disposition(
    document: Mapping[str, Any], repo: Path = ROOT, *,
    reports: Sequence[Mapping[str, Any]] | None = None,
    report_sha256s: Sequence[str] | None = None,
    artifact_sha256s: Sequence[str] | None = None,
) -> dict[str, Any]:
    _schema_validate(repo, "disposition", document)
    candidate = validate_candidate(document["candidate"], repo)
    if document["disposition_id"] != f"rmsnorm-p0-review-{candidate['reviewed_sha']}" or document["tree_oid"] != candidate["git_tree_oid"]:
        raise ContractError("P0 disposition ID/tree is not bound to its candidate")
    expected_identity = {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)}
    expected_policy = {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)}
    if document["matrix"] != expected_identity or document["review_policy"] != expected_policy or document["case_set_sha256"] != case_set_sha256(repo) or document["source_set_sha256"] != source_set(repo)["sha256"]:
        raise ContractError("P0 disposition matrix/policy/case/source identity drifted")
    expected_model = {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION}
    if document["model_lock"] != expected_model or document["threshold"] != expected_review_policy()["threshold"] or document["claims"] != expected_review_policy()["claims"]:
        raise ContractError("P0 disposition threshold/model/claim contract drifted")
    reviewer = document["review"]["reviewer"].strip()
    reason = document["review"]["reason"].strip()
    placeholders = {"tbd", "none", "unknown", "unassigned", "review_required", "n/a"}
    if reviewer.lower() in placeholders or reason.lower() in placeholders or len(reason) < 24:
        raise ContractError("P0 disposition reviewer or reason is missing/placeholder")
    reviewed_at = _parse_time(document["review"]["reviewed_at"], "P0 reviewed_at")
    if reviewed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ContractError("P0 disposition review timestamp is from the future")
    rows = document["canonical_rows"]
    if [row["order"] for row in rows] != [0, 1] or [row["row_id"] for row in rows] != list(ROWS) or [row["target"] for row in rows] != list(TARGETS):
        raise ContractError("P0 disposition canonical rows are missing, duplicate, stale, or reordered")
    for row in rows:
        for label in ("report_sha256", "artifact_sha256", "measurement_sha256"):
            _nonzero_sha(row[label], f"P0 disposition {label}")
        if [case["order"] for case in row["cases"]] != list(range(5)) or [case["id"] for case in row["cases"]] != list(CASE_IDS):
            raise ContractError("P0 disposition case/order set is incomplete")
        for case in row["cases"]:
            for label in ("kernel_median_ns", "wall_median_ns"):
                _strict_positive_ns(case[label], f"P0 disposition {label}")
            for label in ("kernel_mad_ns", "wall_mad_ns"):
                value = case[label]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ContractError(f"P0 disposition {label} is invalid")
            _nonzero_sha(case["sample_set_sha256"], "P0 disposition sample set")
    if reports is not None:
        if len(reports) != 2 or report_sha256s is None or artifact_sha256s is None:
            raise ContractError("P0 disposition comparison lacks exactly two canonical inputs")
        for order, (row, report) in enumerate(zip(rows, reports)):
            if report["candidate"] != candidate or report["target"] != TARGETS[order] or len(report["measurements"]) != 5:
                raise ContractError("P0 disposition references incomplete or mixed reports")
            if row["report_sha256"] != report_sha256s[order] or row["artifact_sha256"] != artifact_sha256s[order] or row["measurement_sha256"] != report["measurement_sha256"] or row["cases"] != _case_summaries(report):
                raise ContractError("P0 disposition does not bind the complete canonical measurements")
            if reviewed_at < _parse_time(report["execution"]["finished_at"], "P0 report finished_at"):
                raise ContractError("P0 disposition predates a canonical measurement")
    return dict(document)


def validate_aggregate(document: Mapping[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    _schema_validate(repo, "aggregate", document)
    candidate = validate_candidate(document["candidate"], repo)
    rows = document["rows"]
    if document["aggregate_id"] != f"rmsnorm-p0-aggregate-{candidate['reviewed_sha']}" or document["tree_oid"] != candidate["git_tree_oid"]:
        raise ContractError("P0 aggregate ID/tree is not bound to its candidate")
    if [row["order"] for row in rows] != [0, 1] or [row["row_id"] for row in rows] != list(ROWS) or [row["target"] for row in rows] != list(TARGETS):
        raise ContractError("P0 aggregate rows are missing, duplicate, mixed, stale, or reordered")
    expected_candidate = candidate_sha256(candidate)
    for row in rows:
        if row["candidate_sha256"] != expected_candidate:
            raise ContractError("P0 aggregate mixes candidate identities")
        for label in ("report_sha256", "artifact_sha256", "tuple_sha256", "measurement_sha256"):
            _nonzero_sha(row[label], f"P0 aggregate {label}")
    if document["matrix"] != {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)} or document["review_policy"] != {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)} or document["case_set_sha256"] != case_set_sha256(repo) or document["source_set_sha256"] != source_set(repo)["sha256"]:
        raise ContractError("P0 aggregate matrix/policy/case/source identity drifted")
    expected_model = {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION}
    if document["model_lock"] != expected_model or document["claims"] != expected_review_policy()["claims"]:
        raise ContractError("P0 aggregate model/claim contract drifted")
    expected_counts = {"expected_rows": 2, "selected_rows": 2, "collected_rows": sum(row["collected_cases"] == 5 for row in rows), "passed_rows": sum(row["state"] == "PASS" for row in rows), "failed_rows": sum(row["state"] == "FAIL" for row in rows), "expected_cases": 10, "collected_cases": sum(row["collected_cases"] for row in rows)}
    if document["counts"] != expected_counts:
        raise ContractError("P0 aggregate counts are inconsistent")
    if document["state"] == "PASS":
        if any(row["state"] != "PASS" or row["collected_cases"] != 5 or row["dispatch_count"] != TOTAL_DISPATCHES or row["fallback_used"] or not row["health_ok"] or not row["process_clean"] for row in rows):
            raise ContractError("P0 aggregate PASS row evidence is incomplete")
    return dict(document)


def validate_contracts(repo: Path = ROOT) -> None:
    validate_matrix(repo)
    validate_review_policy(repo)
    source_set(repo)
    for name in SCHEMAS:
        _schema(repo, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        validate_contracts(args.repo.resolve())
    except (ContractError, OSError, ValueError) as exc:
        print(f"P0 contracts: FAIL: {exc}", file=sys.stderr)
        return 1
    print("P0 contracts: PASS (host validation only; no GPU execution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
