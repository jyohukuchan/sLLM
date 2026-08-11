#!/usr/bin/env python3
"""Fail-closed aggregation for the two non-required Phase 2 H3 rows.

The H3 workflow deliberately has a contract separate from the required host
workflow.  This module is stdlib-only so the final aggregation job does not
need to install a schema or test dependency.  It validates the checked-in
toolchain/matrix, each downloaded row directory, all four content sidecars,
and the cross-row immutable identity before emitting ``aggregate.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TARGETS = ("gfx1030", "gfx1201")
EXPECTED_ROWS = tuple(f"h3-{target}" for target in EXPECTED_TARGETS)
PINNED_IMAGE_REFERENCE = (
    "docker.io/rocm/dev-ubuntu-24.04@"
    "sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
)
EXPECTED_MANIFEST_DIGEST = "sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
EXPECTED_CONFIG_DIGEST = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
EXPECTED_CODEGEN_FEATURES = {
    "xnack": "unsupported",
    "sramecc": "unsupported",
    "generic_processor_version": 0,
}
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
EXPECTED_SCOPE = {
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
}
EXPECTED_ENVIRONMENT = {
    "mode": "required-ci",
    "execution_scope": "official-container",
    "container_image_reference": PINNED_IMAGE_REFERENCE,
    "observed_image_config_digest": EXPECTED_CONFIG_DIGEST,
    "pinned_container": True,
    "identity_verified": True,
    "network_isolated": True,
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
REPORT_KEYS = {
    "schema_version", "result_id", "suite_id", "tier", "state", "required",
    "evidence_mode", "run_id", "run_attempt", "reviewed_sha", "tested_sha",
    "workflow_sha", "git_tree_oid", "worktree_clean",
    "matrix_manifest_sha256", "matrix_row_id", "tuple_digest", "command",
    "command_sha256", "toolchain", "toolchain_sha256", "artifact",
    "h3_artifact", "h3_scope", "execution_environment", "created_at",
    "started_at", "finished_at", "duration_seconds", "seed", "counts",
    "resource", "cases", "steps", "diagnostic",
}
METADATA_KEYS = {
    "schema_version", "metadata_id", "matrix_row_id", "target", "candidate",
    "toolchain_id", "matrix_id", "toolchain_manifest_sha256",
    "matrix_manifest_sha256", "image", "resolved_paths", "build", "codegen",
    "artifact", "host_bundle", "device_code_object", "scope",
    "execution_environment", "timestamps", "duration_seconds",
}
REPORT_RESOURCE_KEYS = {
    "wall_time_limit_seconds", "wall_time_breach", "max_rss_bytes",
    "max_rss_limit_bytes", "rss_breach", "runner_max_rss_bytes",
    "fixture_size_bytes", "fixture_size_limit_bytes", "fixture_size_breach",
    "output_bytes", "captured_output_bytes", "row_output_limit_bytes",
    "output_breach", "address_space_limit_bytes", "commands_expected",
    "commands_executed", "commands_complete", "network_isolated",
    "network_guard_strategies",
}
STEP_KEYS = {
    "step_id", "state", "started_at", "finished_at", "duration_seconds",
    "exit_code", "stdout_sha256", "stderr_sha256", "diagnostic",
    "selection_required", "count_source", "counts", "resource",
}
STEP_RESOURCE_KEYS = {
    "wall_time_limit_seconds", "timed_out", "max_rss_bytes",
    "max_rss_limit_bytes", "rss_breach", "cpu_user_seconds",
    "cpu_system_seconds", "stdout_bytes", "stderr_bytes", "output_bytes",
    "stdout_captured_bytes", "stderr_captured_bytes", "captured_output_bytes",
    "output_limit_bytes", "output_breach", "network_isolated",
    "network_guard_strategy", "address_space_limit_bytes",
    "address_space_limit_enforced",
}


class ContractError(ValueError):
    """A malformed H3 input; callers must fail closed."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def render_direct_commands(row: dict[str, Any], target: str, output_directory: str, source_path: str) -> list[list[str]]:
    """Reconstruct the only two H3 compiler commands from the checked-in row."""

    replacements = {
        "{target}": target,
        "{build_dir}": output_directory,
        "{source_path}": source_path,
    }
    commands: list[list[str]] = []
    for template in row["direct_build"]["commands"]:
        command = []
        for token in template:
            for placeholder, value in replacements.items():
                token = token.replace(placeholder, value)
            if "{" in token or "}" in token:
                raise ContractError("direct H3 command has an unresolved template placeholder")
            command.append(token)
        commands.append(command)
    return commands


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs_without_duplicates)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON document is not an object: {path}")
    return value


def exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ContractError(f"{label} does not match the H3 contract")


def exact_sha(value: Any, label: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise ContractError(f"{label} is not a 40-character lowercase SHA")
    if expected is not None and value != expected:
        raise ContractError(f"{label} is stale or mismatched")
    return value


def exact_hash(value: Any, label: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{label} is not a 64-character lowercase SHA-256")
    if expected is not None and value != expected:
        raise ContractError(f"{label} is stale or mismatched")
    return value


def require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{label} has missing or unknown fields")
    return value


def require_nonnegative_number(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label} is not a non-negative number")


def git_identity(repo: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise ContractError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    exact_sha(commit, "checked-out commit")
    exact_sha(tree, "checked-out tree OID")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ContractError("strict H3 aggregation rejects a dirty checkout")
    return {"commit": commit, "tree": tree}


def load_contract(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    toolchain = read_json(repo / "ci/toolchains/rocm-7.14.0.json")
    matrix = read_json(repo / "ci/matrix/hip-compile-v1.json")
    exact(toolchain.get("schema_version"), "rocm-toolchain-v1", "toolchain schema version")
    exact(toolchain.get("toolchain_id"), "rocm-7.14.0", "toolchain id")
    image = toolchain.get("image")
    if not isinstance(image, dict):
        raise ContractError("toolchain image is missing")
    exact(image.get("repository"), "docker.io/rocm/dev-ubuntu-24.04", "toolchain image repository")
    exact(image.get("tag"), "7.14.0-full", "toolchain image tag")
    exact(image.get("manifest_digest"), EXPECTED_MANIFEST_DIGEST, "toolchain image manifest digest")
    exact(image.get("config_digest"), EXPECTED_CONFIG_DIGEST, "toolchain image config digest")
    exact(image.get("manifest_list_digest"), None, "toolchain image manifest-list digest")
    exact(image.get("platform"), {"os": "linux", "architecture": "amd64"}, "toolchain image platform")
    exact(toolchain.get("rocm"), {"path": "/opt/rocm", "version": "7.14.0", "llvm_major": 23}, "ROCm tuple")
    paths = toolchain.get("paths")
    if not isinstance(paths, dict) or paths.get("rocm_root") != "/opt/rocm" or paths.get("compiler") != "/opt/rocm/bin/amdclang++":
        raise ContractError("toolchain paths are not bound to /opt/rocm")
    exact(matrix.get("schema_version"), "hip-compile-v1", "matrix schema version")
    exact(matrix.get("matrix_id"), "hip-compile-v1", "matrix id")
    exact(matrix.get("toolchain_id"), "rocm-7.14.0", "matrix toolchain id")
    exact(matrix.get("targets"), list(EXPECTED_TARGETS), "matrix target set")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ROWS):
        raise ContractError("H3 matrix must contain exactly two rows")
    by_row: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("H3 matrix row is not an object")
        row_id = row.get("row_id")
        target = row.get("target")
        if row_id not in EXPECTED_ROWS or target not in EXPECTED_TARGETS or row_id != f"h3-{target}" or row_id in by_row:
            raise ContractError("H3 matrix has missing, duplicate, or unknown rows")
        exact(row.get("tier"), "tier_h3", f"{row_id} tier")
        exact(row.get("required"), False, f"{row_id} required flag")
        exact(row.get("execution"), {"mode": "compile-only", "requires_gpu": False, "requires_model": False, "network": False, "fallback_allowed": False}, f"{row_id} execution scope")
        exact(row.get("direct_build"), DIRECT_BUILD, f"{row_id} direct amdclang++ contract")
        exact(row.get("codegen", {}).get("target"), target, f"{row_id} codegen target")
        exact(row.get("codegen", {}).get("target_kind"), "exact", f"{row_id} codegen kind")
        exact(row.get("codegen", {}).get("target_count"), 1, f"{row_id} codegen count")
        exact(row.get("codegen", {}).get("code_object_version"), "V6", f"{row_id} code object version")
        exact(row.get("codegen", {}).get("wavefront_size"), 32, f"{row_id} wavefront size")
        exact(row.get("codegen", {}).get("features"), EXPECTED_CODEGEN_FEATURES, f"{row_id} codegen features")
        by_row[row_id] = row
    if set(by_row) != set(EXPECTED_ROWS):
        raise ContractError("H3 matrix is not exactly gfx1030/gfx1201")
    return toolchain, matrix, by_row


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{label} timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def require_regular(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} is missing or is not a regular non-symlink file")


def validate_sidecar(path: Path, target: Path, label: str) -> str:
    require_regular(path, label)
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{label} is not an ASCII sidecar: {exc}") from exc
    expected_name = target.name
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", value)
    if not match or match.group(2) != expected_name:
        raise ContractError(f"{label} has a malformed or wrong-name sidecar")
    digest = sha256_file(target)
    if match.group(1) != digest:
        raise ContractError(f"{label} does not match {expected_name}")
    return sha256_file(path)


def validate_artifact_fields(metadata: dict[str, Any], target: str, artifact: Path, metadata_bytes: bytes, expected: dict[str, Any]) -> None:
    require_keys(metadata, METADATA_KEYS, f"{target} metadata")
    artifact_record = metadata.get("artifact")
    artifact_record = require_keys(
        metadata.get("artifact"), {"path", "size_bytes", "sha256"},
        f"{target} metadata artifact record",
    )
    exact(artifact_record.get("size_bytes"), artifact.stat().st_size, f"{target} artifact size")
    exact_hash(artifact_record.get("sha256"), f"{target} artifact metadata SHA", sha256_file(artifact))
    artifact_path = artifact_record.get("path")
    if not isinstance(artifact_path, str):
        raise ContractError(f"{target}: artifact path is missing")
    parsed = PurePosixPath(artifact_path)
    if not parsed.is_absolute() or parsed.name != f"device-code-object-{target}.elf" or parsed.parent.name != f"h3-{target}":
        raise ContractError(f"{target}: artifact path is not private to its exact target")
    if any(other in parsed.name for other in ("gfx1030", "gfx1201") if other != target):
        raise ContractError(f"{target}: artifact path contains another H3 target")
    build = metadata.get("build")
    build = require_keys(
        build,
        {
            "source_directory", "source_path", "output_directory", "object_path", "link_output_path",
            "generator", "mode", "build_type", "language_standard",
            "output_directory_scope", "source_tree_output", "shared_build_directory",
        },
        f"{target} metadata build record",
    )
    exact(build.get("source_directory"), "/workspace", f"{target} source directory")
    exact(build.get("source_path"), "/workspace/native/hip/src/hip_compile_probe.hip.cpp", f"{target} source path")
    output_directory = build.get("output_directory")
    if not isinstance(output_directory, str):
        raise ContractError(f"{target}: output directory is missing")
    output = PurePosixPath(output_directory)
    if not output.is_absolute() or output.name != f"h3-{target}" or parsed.parent != output:
        raise ContractError(f"{target}: output directory is not row-private")
    exact(build.get("object_path"), f"{output_directory}/hip-compile-probe-{target}.o", f"{target} direct object output")
    exact(build.get("link_output_path"), f"{output_directory}/hip-compile-probe-{target}.elf", f"{target} direct link output")
    exact(build.get("generator"), "direct-amdclang++", f"{target} direct compiler generator")
    exact(build.get("mode"), "direct-compile-link", f"{target} direct build mode")
    exact(build.get("build_type"), "Release", f"{target} build type")
    exact(build.get("language_standard"), "gnu++17", f"{target} language standard")
    exact(build.get("output_directory_scope"), "row-private", f"{target} output scope")
    exact(build.get("source_tree_output"), False, f"{target} source-tree output flag")
    exact(build.get("shared_build_directory"), False, f"{target} shared build flag")
    exact(metadata.get("scope"), EXPECTED_SCOPE, f"{target} metadata compile-only scope")
    environment = metadata.get("execution_environment")
    exact(environment, EXPECTED_ENVIRONMENT, f"{target} metadata execution environment")
    host = metadata.get("host_bundle")
    if not isinstance(host, dict) or set(host) != {"format", "machine", "bundles", "sections"}:
        raise ContractError(f"{target}: host ELF evidence has missing or unknown fields")
    exact(host, {
        "format": "ELF64",
        "machine": "X86_64",
        "bundles": [
            {"id": f"hipv4-amdgcn-amd-amdhsa--{target}", "target": target},
            {"id": "host-x86_64-unknown-linux-gnu-", "target": "host"},
        ],
        "sections": host.get("sections") if isinstance(host, dict) else None,
    }, f"{target} host ELF identity")
    if set(host["sections"]) - {".text", ".hip_fatbin"} or ".hip_fatbin" not in host["sections"]:
        raise ContractError(f"{target}: host ELF sections are not closed")
    if not all(isinstance(section, dict) and set(section) == {"present", "size_bytes"} and section["present"] is True and isinstance(section["size_bytes"], int) and section["size_bytes"] >= 0 for section in host["sections"].values()):
        raise ContractError(f"{target}: host ELF section evidence is invalid")
    if not host["sections"][".hip_fatbin"]["present"]:
        raise ContractError(f"{target}: host bundle does not prove .hip_fatbin")
    device = metadata.get("device_code_object")
    device = require_keys(
        device,
        {
            "format", "machine", "target", "ei_abiversion", "e_flags",
            "code_object_version", "wavefront_size", "features", "sections",
            "symbols",
        },
        f"{target} device ELF evidence",
    )
    exact(device.get("format"), "ELF64", f"{target} device ELF format")
    exact(device.get("machine"), "AMDGPU", f"{target} device ELF machine")
    exact(device.get("target"), target, f"{target} device ELF target")
    exact(device.get("ei_abiversion"), 4, f"{target} device ABI")
    expected_flags = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}[target]
    exact(device.get("e_flags"), expected_flags, f"{target} device e_flags")
    exact(device.get("code_object_version"), "V6", f"{target} device code object version")
    exact(device.get("wavefront_size"), 32, f"{target} device wavefront")
    exact(device.get("features"), EXPECTED_CODEGEN_FEATURES, f"{target} device features")
    if set(device["sections"]) != {".text"} or not isinstance(device["sections"][".text"], dict) or set(device["sections"][".text"]) != {"present", "size_bytes"} or device["sections"][".text"]["present"] is not True or not isinstance(device["sections"][".text"]["size_bytes"], int) or device["sections"][".text"]["size_bytes"] < 0:
        raise ContractError(f"{target}: device ELF does not prove .text")
    symbols = device.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(not isinstance(symbol, dict) or set(symbol) != {"name", "defined"} or not isinstance(symbol["name"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol["name"]) or symbol["defined"] is not True for symbol in symbols) or not any(symbol["name"] == "sllm_hip_compile_probe" for symbol in symbols):
        raise ContractError(f"{target}: device ELF has no defined compile-probe symbol")
    for name in ("created_at", "started_at", "finished_at"):
        parse_time(metadata.get("timestamps", {}).get(name), f"{target} metadata {name}")
    timestamps = metadata["timestamps"]
    if not (parse_time(timestamps["created_at"], "created_at") <= parse_time(timestamps["started_at"], "started_at") <= parse_time(timestamps["finished_at"], "finished_at")):
        raise ContractError(f"{target}: metadata timestamps are not ordered")
    if not isinstance(metadata.get("duration_seconds"), (int, float)) or metadata["duration_seconds"] < 0:
        raise ContractError(f"{target}: metadata duration is invalid")


def validate_report_shape(report: dict[str, Any], row_id: str, expected: dict[str, Any]) -> None:
    require_keys(report, REPORT_KEYS, f"{row_id} report")
    toolchain = require_keys(
        report["toolchain"],
        {
            "python", "platform", "system", "machine", "git", "rustc_dev",
            "cargo_dev", "rustc_msrv", "cargo_msrv", "clang_format", "cmake",
            "host_packages", "h3",
        },
        f"{row_id} report toolchain",
    )
    if any(not isinstance(toolchain[key], str) or not toolchain[key] for key in ("python", "platform", "system", "machine", "git", "rustc_dev", "cargo_dev", "rustc_msrv", "cargo_msrv", "clang_format", "cmake")):
        raise ContractError(f"{row_id}: report host toolchain observation is invalid")
    exact(toolchain["cmake"], "not-applicable", f"{row_id} CMake observation")
    exact(toolchain["host_packages"], {"h3": "compile-only"}, f"{row_id} report host packages")
    resource_record = require_keys(report["resource"], REPORT_RESOURCE_KEYS, f"{row_id} report resource")
    for key in ("wall_time_limit_seconds", "max_rss_bytes", "max_rss_limit_bytes", "runner_max_rss_bytes", "fixture_size_bytes", "fixture_size_limit_bytes", "output_bytes", "captured_output_bytes", "row_output_limit_bytes", "commands_expected", "commands_executed"):
        require_nonnegative_number(resource_record[key], f"{row_id} report resource.{key}")
    if resource_record["wall_time_limit_seconds"] <= 0 or resource_record["max_rss_limit_bytes"] != expected["resource"]["max_rss_bytes"] or resource_record["row_output_limit_bytes"] != expected["resource"]["max_output_bytes"] or resource_record["fixture_size_limit_bytes"] <= 0 or resource_record["address_space_limit_bytes"] != expected["resource"]["max_rss_bytes"]:
        raise ContractError(f"{row_id}: report resource limits are not the H3 row limits")
    if any(resource_record[key] is not False for key in ("wall_time_breach", "rss_breach", "fixture_size_breach", "output_breach")) or resource_record["commands_expected"] != 2 or resource_record["commands_executed"] != 2 or resource_record["commands_complete"] is not True or resource_record["network_isolated"] is not True or resource_record["network_guard_strategies"] != ["container-network-none"]:
        raise ContractError(f"{row_id}: report resource claims do not prove a clean H3 PASS")
    if report["cases"] != []:
        raise ContractError(f"{row_id}: H3 report must not record executable test cases")
    for index, step in enumerate(report["steps"], start=1):
        step = require_keys(step, STEP_KEYS, f"{row_id} step {index}")
        step_resource = require_keys(step["resource"], STEP_RESOURCE_KEYS, f"{row_id} step {index} resource")
        if step["step_id"] != f"{row_id}.command-{index}" or step["state"] != "PASS" or step["selection_required"] is not True or step["count_source"] != "validator-command":
            raise ContractError(f"{row_id}: step {index} is not a required compile/link PASS")
        exact(step["counts"], {"collected": 1, "selected": 1, "passed": 1, "failed": 0, "skipped": 0, "deselected": 0}, f"{row_id} step {index} counts")
        if not isinstance(step["exit_code"], int) or step["exit_code"] != 0 or not isinstance(step["diagnostic"], str) or step_resource["timed_out"] is not False or step_resource["rss_breach"] is not False or step_resource["output_breach"] is not False or step_resource["network_isolated"] is not True or step_resource["network_guard_strategy"] != "container-network-none" or step_resource["address_space_limit_bytes"] != expected["resource"]["max_rss_bytes"] or step_resource["address_space_limit_enforced"] is not True:
            raise ContractError(f"{row_id}: step {index} resource claims are invalid")
        for name in ("started_at", "finished_at"):
            parse_time(step[name], f"{row_id} step {index} {name}")
        for name in ("duration_seconds", "wall_time_limit_seconds", "max_rss_bytes", "max_rss_limit_bytes", "cpu_user_seconds", "cpu_system_seconds", "stdout_bytes", "stderr_bytes", "output_bytes", "stdout_captured_bytes", "stderr_captured_bytes", "captured_output_bytes", "output_limit_bytes"):
            require_nonnegative_number(step_resource[name], f"{row_id} step {index} resource.{name}")
        if step_resource["wall_time_limit_seconds"] <= 0 or step_resource["max_rss_limit_bytes"] != expected["resource"]["max_rss_bytes"] or step_resource["output_limit_bytes"] != expected["resource"]["max_output_bytes"]:
            raise ContractError(f"{row_id}: step {index} resource limit is invalid")
    diagnostic = report["diagnostic"]
    if not isinstance(diagnostic, dict) or set(diagnostic) not in ({"message", "errors", "warnings", "network_disabled", "model_disabled", "gpu_fallback_disabled", "network_guard_self_test"}, {"message", "errors", "warnings", "output_dir", "network_disabled", "model_disabled", "gpu_fallback_disabled", "network_guard_self_test"}) or not isinstance(diagnostic["message"], str) or not isinstance(diagnostic["warnings"], list) or diagnostic["errors"] != [] or diagnostic["network_disabled"] is not True or diagnostic["model_disabled"] is not True or diagnostic["gpu_fallback_disabled"] is not True or diagnostic["network_guard_self_test"] is not True:
        raise ContractError(f"{row_id}: report diagnostic is invalid")


def validate_row(row_dir: Path, row_id: str, expected: dict[str, Any], toolchain: dict[str, Any], matrix: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    target = expected["target"]
    if row_dir.name != row_id or not row_dir.is_dir() or row_dir.is_symlink():
        raise ContractError(f"{row_id}: row directory is missing or not private")
    artifact_name = f"device-code-object-{target}.elf"
    expected_files = {"report.json", "report.json.sha256", "hip-artifact-metadata.json", "hip-artifact-metadata.json.sha256", artifact_name, f"{artifact_name}.sha256"}
    actual_files = {path.name for path in row_dir.iterdir()}
    if actual_files != expected_files:
        raise ContractError(f"{row_id}: missing, duplicate, or unknown row artifact files")
    paths = {name: row_dir / name for name in expected_files}
    for path in paths.values():
        require_regular(path, f"{row_id}: {path.name}")
    report_path = paths["report.json"]
    metadata_path = paths["hip-artifact-metadata.json"]
    artifact_path = paths[artifact_name]
    report_sidecar_sha = validate_sidecar(paths["report.json.sha256"], report_path, f"{row_id}: report sidecar")
    metadata_sidecar_sha = validate_sidecar(paths["hip-artifact-metadata.json.sha256"], metadata_path, f"{row_id}: metadata sidecar")
    artifact_sidecar_sha = validate_sidecar(paths[f"{artifact_name}.sha256"], artifact_path, f"{row_id}: artifact sidecar")
    report_bytes = report_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    report = read_json(report_path)
    metadata = read_json(metadata_path)
    exact(report.get("schema_version"), "test-result-v1", f"{row_id} report schema")
    exact(report.get("result_id"), f"{row_id}.{identity['run_id']}.{identity['run_attempt']}", f"{row_id} result id")
    exact(report.get("suite_id"), row_id, f"{row_id} suite id")
    exact(report.get("matrix_row_id"), row_id, f"{row_id} matrix row id")
    exact(report.get("tier"), "tier_h3", f"{row_id} report tier")
    exact(report.get("state"), "PASS", f"{row_id} report state")
    exact(report.get("required"), False, f"{row_id} report required flag")
    exact(report.get("evidence_mode"), "required-ci", f"{row_id} evidence mode")
    for key in ("run_id", "run_attempt", "reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid"):
        exact(report.get(key), identity[key], f"{row_id} report {key}")
    exact(report.get("matrix_manifest_sha256"), sha256_json(matrix), f"{row_id} report matrix hash")
    exact(report.get("seed"), expected["seed"], f"{row_id} seed")
    exact(report.get("tuple_digest"), sha256_json(expected), f"{row_id} tuple digest")
    exact(report.get("worktree_clean"), True, f"{row_id} clean checkout flag")
    exact(report.get("h3_scope"), EXPECTED_SCOPE, f"{row_id} report compile-only scope")
    exact(report.get("execution_environment"), EXPECTED_ENVIRONMENT, f"{row_id} report execution environment")
    artifact_record = report.get("artifact")
    exact(artifact_record, {"content_sha256": sha256_file(artifact_path), "manifest_sha256": sha256_bytes(metadata_bytes)}, f"{row_id} report artifact record")
    h3_artifact = report.get("h3_artifact")
    exact(h3_artifact, {
        "target": target,
        "size_bytes": artifact_path.stat().st_size,
        "content_sha256": sha256_file(artifact_path),
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "metadata_sidecar_sha256": metadata_sidecar_sha,
        "artifact_sidecar_sha256": artifact_sidecar_sha,
    }, f"{row_id} report artifact evidence")
    toolchain_report = report.get("toolchain")
    if not isinstance(toolchain_report, dict) or not isinstance(toolchain_report.get("h3"), dict):
        raise ContractError(f"{row_id}: report H3 toolchain observation is missing")
    exact(toolchain_report.get("h3", {}).get("toolchain_id"), toolchain["toolchain_id"], f"{row_id} report toolchain id")
    exact(toolchain_report["h3"].get("manifest_sha256"), sha256_json(toolchain), f"{row_id} report toolchain hash")
    exact(report.get("toolchain_sha256"), sha256_json(toolchain_report), f"{row_id} report toolchain record hash")
    exact(report.get("diagnostic", {}).get("errors"), [], f"{row_id} report diagnostics")
    exact(report.get("diagnostic", {}).get("network_disabled"), True, f"{row_id} network disabled marker")
    exact(report.get("diagnostic", {}).get("model_disabled"), True, f"{row_id} model disabled marker")
    exact(report.get("diagnostic", {}).get("gpu_fallback_disabled"), True, f"{row_id} fallback disabled marker")
    exact(report.get("diagnostic", {}).get("network_guard_self_test"), True, f"{row_id} network guard marker")
    counts = report.get("counts")
    exact(counts, {"collected": 2, "selected": 2, "passed": 2, "failed": 0, "skipped": 0, "deselected": 0}, f"{row_id} report counts")
    steps = report.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise ContractError(f"{row_id}: report does not contain exactly two compile/link steps")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or step.get("step_id") != f"{row_id}.command-{index}" or step.get("state") != "PASS" or step.get("selection_required") is not True:
            raise ContractError(f"{row_id}: step {index} is not a required PASS")
        if step.get("resource", {}).get("network_isolated") is not True or step.get("resource", {}).get("network_guard_strategy") != "container-network-none":
            raise ContractError(f"{row_id}: step {index} lacks the required network boundary")
    commands = report.get("command")
    if not isinstance(commands, list) or len(commands) != 2 or any(not isinstance(command, list) for command in commands):
        raise ContractError(f"{row_id}: report command list is not exactly two direct compiler commands")
    build = metadata.get("build")
    if not isinstance(build, dict):
        raise ContractError(f"{row_id}: metadata build record is missing")
    expected_commands = render_direct_commands(expected, target, build.get("output_directory", ""), build.get("source_path", ""))
    exact(commands, expected_commands, f"{row_id} exact direct compile/link commands")
    exact(report.get("command_sha256"), sha256_json(commands), f"{row_id} command hash")
    for name in ("created_at", "started_at", "finished_at"):
        parse_time(report.get(name), f"{row_id} report {name}")
    if not (parse_time(report["created_at"], "created_at") <= parse_time(report["started_at"], "started_at") <= parse_time(report["finished_at"], "finished_at")):
        raise ContractError(f"{row_id}: report timestamps are not ordered")
    if report.get("finished_at") and parse_time(report["finished_at"], "finished_at") > datetime.now(timezone.utc):
        raise ContractError(f"{row_id}: report timestamp is in the future")
    exact(metadata.get("schema_version"), "hip-artifact-metadata-v1", f"{row_id} metadata schema")
    exact(metadata.get("metadata_id"), f"h3-artifact-{target}", f"{row_id} metadata id")
    exact(metadata.get("matrix_row_id"), row_id, f"{row_id} metadata row id")
    exact(metadata.get("target"), target, f"{row_id} metadata target")
    candidate = metadata.get("candidate")
    if not isinstance(candidate, dict):
        raise ContractError(f"{row_id}: metadata candidate is missing")
    for key in ("commit_sha", "reviewed_sha", "tested_sha", "workflow_sha"):
        exact(candidate.get(key), identity["reviewed_sha"], f"{row_id} metadata {key}")
    exact(candidate.get("tree_oid"), identity["git_tree_oid"], f"{row_id} metadata tree OID")
    exact(metadata.get("toolchain_id"), toolchain["toolchain_id"], f"{row_id} metadata toolchain id")
    exact(metadata.get("matrix_id"), matrix["matrix_id"], f"{row_id} metadata matrix id")
    exact(metadata.get("toolchain_manifest_sha256"), sha256_json(toolchain), f"{row_id} metadata toolchain hash")
    exact(metadata.get("matrix_manifest_sha256"), sha256_json(matrix), f"{row_id} metadata matrix hash")
    image_keys = ("repository", "tag", "manifest_digest", "config_digest", "manifest_list_digest", "manifest_type", "platform")
    exact(metadata.get("image"), {key: toolchain["image"][key] for key in image_keys}, f"{row_id} metadata image")
    exact(metadata.get("resolved_paths"), toolchain["paths"], f"{row_id} metadata resolved paths")
    exact(metadata.get("codegen"), expected["codegen"], f"{row_id} metadata codegen")
    validate_artifact_fields(metadata, target, artifact_path, metadata_bytes, expected)
    return {
        "row_id": row_id,
        "target": target,
        "state": "PASS",
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
        "report_sidecar_sha256": report_sidecar_sha,
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_sidecar_sha256": metadata_sidecar_sha,
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_sidecar_sha256": artifact_sidecar_sha,
    }


def load_needs(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if set(value) != set(EXPECTED_ROWS):
        raise ContractError("needs JSON has missing or unknown H3 jobs")
    for row_id, entry in value.items():
        if not isinstance(entry, dict) or entry.get("result") != "success":
            raise ContractError(f"needs.{row_id} is not success")
    return value


def aggregate_results(*, needs_path: Path, artifact_dir: Path, repo: Path, output_dir: Path, run_id: str, run_attempt: int, reviewed_sha: str, tested_sha: str, workflow_sha: str, tree_oid: str | None = None) -> dict[str, Any]:
    if not RUN_ID.fullmatch(run_id) or run_attempt < 1:
        raise ContractError("run identity is invalid")
    for value, name in ((reviewed_sha, "reviewed_sha"), (tested_sha, "tested_sha"), (workflow_sha, "workflow_sha")):
        exact_sha(value, name)
    if len({reviewed_sha, tested_sha, workflow_sha}) != 1:
        raise ContractError("reviewed/tested/workflow SHA values must be identical")
    identity = git_identity(repo)
    expected_identity = {"run_id": run_id, "run_attempt": run_attempt, "reviewed_sha": reviewed_sha, "tested_sha": tested_sha, "workflow_sha": workflow_sha, "git_tree_oid": tree_oid or identity["tree"]}
    exact(expected_identity["reviewed_sha"], identity["commit"], "reviewed SHA")
    exact(expected_identity["git_tree_oid"], identity["tree"], "tree OID")
    load_needs(needs_path)
    toolchain, matrix, rows = load_contract(repo)
    if not artifact_dir.is_dir():
        raise ContractError("H3 artifact directory is missing")
    children = {path.name for path in artifact_dir.iterdir()}
    if children != set(EXPECTED_ROWS):
        raise ContractError("H3 artifact directory has missing or unknown row directories")
    summaries = [validate_row(artifact_dir / row_id, row_id, rows[row_id], toolchain, matrix, expected_identity) for row_id in EXPECTED_ROWS]
    return {
        "schema_version": "h3-aggregate-v1",
        "aggregate_id": f"h3-aggregate.{run_id}.{run_attempt}",
        "state": "PASS",
        "required": False,
        "evidence_mode": "required-ci",
        "run_id": run_id,
        "run_attempt": run_attempt,
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": identity["tree"],
        "toolchain_id": toolchain["toolchain_id"],
        "toolchain_manifest_sha256": sha256_json(toolchain),
        "matrix_id": matrix["matrix_id"],
        "matrix_manifest_sha256": sha256_json(matrix),
        "expected_rows": list(EXPECTED_ROWS),
        "scope": {"compile_only": True, "gpu_execution": False, "execution_attempted": False, "model_used": False, "cpu_fallback_used": False},
        "rows": summaries,
        "errors": [],
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "aggregate.json"
    data = canonical_bytes(summary)
    report.write_bytes(data)
    report.with_name(report.name + ".sha256").write_text(f"{sha256_bytes(data)}  {report.name}\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--needs-json", type=Path, required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--expected-reviewed-sha", "--reviewed-sha", dest="reviewed_sha", required=True)
    result.add_argument("--expected-tested-sha", "--tested-sha", dest="tested_sha", required=True)
    result.add_argument("--expected-workflow-sha", "--workflow-sha", dest="workflow_sha", required=True)
    result.add_argument("--expected-tree-oid", "--tree-oid", dest="tree_oid")
    result.add_argument("--strict-ci", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary: dict[str, Any]
    try:
        if not args.strict_ci:
            raise ContractError("H3 aggregation requires --strict-ci")
        summary = aggregate_results(
            needs_path=args.needs_json,
            artifact_dir=args.artifact_dir,
            repo=args.repo.resolve(),
            output_dir=args.output_dir.resolve(),
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            reviewed_sha=args.reviewed_sha,
            tested_sha=args.tested_sha,
            workflow_sha=args.workflow_sha,
            tree_oid=args.tree_oid,
        )
    except (ContractError, OSError, TypeError, ValueError) as exc:
        summary = {
            "schema_version": "h3-aggregate-v1",
            "aggregate_id": f"h3-aggregate.{args.run_id}.{args.run_attempt}",
            "state": "FAIL",
            "required": False,
            "evidence_mode": "required-ci",
            "run_id": args.run_id,
            "run_attempt": args.run_attempt,
            "reviewed_sha": args.reviewed_sha,
            "tested_sha": args.tested_sha,
            "workflow_sha": args.workflow_sha,
            "expected_rows": list(EXPECTED_ROWS),
            "rows": [],
            "errors": [str(exc)],
        }
        print(f"H3 aggregate: FAIL: {exc}", file=sys.stderr)
        try:
            write_summary(args.output_dir.resolve(), summary)
        except OSError as write_error:
            print(f"H3 aggregate: cannot write failure summary: {write_error}", file=sys.stderr)
        return 1
    write_summary(args.output_dir.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
