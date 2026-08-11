#!/usr/bin/env python3
"""Validate the checked-in H3 public-runtime contracts and optional artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from run_h3_public_runtime_compile import (
        E_FLAGS, EXPECTED_DIRECT_COMPILE_SOURCE_PATHS, EXPECTED_FEATURES, EXPECTED_SOURCE_PATHS, PINNED_CONFIG,
        PINNED_IMAGE, PUBLIC_SYMBOLS, ROOT, SHA40, SHA256, TARGETS,
        RuntimeContractError, canonical_bytes, expected_build_commands, read_json, sha256_file, sha256_json,
        validate_matrix,
    )
except ImportError:  # pragma: no cover - package import path used by some test runners
    from ci.tools.run_h3_public_runtime_compile import (
        E_FLAGS, EXPECTED_DIRECT_COMPILE_SOURCE_PATHS, EXPECTED_FEATURES, EXPECTED_SOURCE_PATHS, PINNED_CONFIG,
        PINNED_IMAGE, PUBLIC_SYMBOLS, ROOT, SHA40, SHA256, TARGETS,
        RuntimeContractError, canonical_bytes, expected_build_commands, read_json, sha256_file, sha256_json,
        validate_matrix,
    )

ARTIFACT_SCHEMA = "ci/schema/hip-runtime-artifact-v1.schema.json"
REPORT_SCHEMA = "ci/schema/hip-runtime-public-report-v1.schema.json"
COMPILE_SCHEMA = "ci/schema/hip-runtime-compile-v1.schema.json"
AGGREGATE_SCHEMA = "ci/schema/hip-runtime-aggregate-v1.schema.json"
EXPECTED_SCOPE = {"public_runtime_stub_linked": False, "compile_only": True, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "support_claim": False, "numerics_verified": False, "performance_verified": False}
EXPECTED_ENVIRONMENT = {"mode": "required-ci", "execution_scope": "official-container", "container_image_reference": PINNED_IMAGE, "observed_image_config_digest": PINNED_CONFIG, "pinned_container": True, "identity_verified": True, "network_isolated": True}
EXPECTED_OUTPUT_HASH_KEYS = {"probe_object", "public_runtime_object", "rmsnorm_kernel_object", "rmsnorm_api_object", "host_elf", "probe_fatbin", "device_object"}
EXPECTED_REPORT_HASH_KEYS = EXPECTED_OUTPUT_HASH_KEYS | {"metadata"}
EXPECTED_STEP_MAX_RSS_LIMIT_BYTES = 4294967296
EXPECTED_STEP_OUTPUT_LIMIT_BYTES = 16777216


class ContractError(ValueError):
    pass


ARTIFACT_METADATA_NAME = "hip-runtime-artifact.json"
EXPECTED_ROW_IDS = {f"h3-public-{target}" for target in TARGETS}


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving symlinks."""

    return Path(os.path.abspath(path))


def _has_symlink_component(path: Path) -> bool:
    """Return whether any component of an absolute path is a symlink."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _collection_root(path: Path) -> Path:
    root = _absolute_lexical(path)
    if _has_symlink_component(root) or not root.is_dir():
        raise ContractError("strict artifact validation requires a non-symlink collection root")
    return root


def _validate_collection_entries(root: Path) -> list[Path]:
    """Require every collection-root child to be one direct, real row directory."""

    entries = sorted(root.iterdir(), key=lambda path: path.name)
    unexpected = [entry.name for entry in entries if entry.name not in EXPECTED_ROW_IDS]
    if unexpected:
        raise ContractError(f"collection root has unexpected direct entries: {unexpected}")
    invalid = [
        entry.name
        for entry in entries
        if entry.name in EXPECTED_ROW_IDS and (entry.is_symlink() or not entry.is_dir())
    ]
    if invalid:
        raise ContractError(f"collection root row entries are not direct directories: {invalid}")
    return entries


def _discover_metadata_paths(artifact_root: Path) -> list[Path]:
    """Discover only direct row children of an exact collection root."""

    root = _collection_root(artifact_root)
    return [row_dir / ARTIFACT_METADATA_NAME for row_dir in _validate_collection_entries(root)]


def _row_roots_for_collection(
    metadata_paths: Iterable[Path], artifact_root: Path
) -> list[tuple[Path, Path]]:
    root = _collection_root(artifact_root)
    _validate_collection_entries(root)
    seen: set[Path] = set()
    result: list[tuple[Path, Path]] = []
    for supplied_path in metadata_paths:
        metadata_path = _absolute_lexical(Path(supplied_path))
        if metadata_path in seen:
            raise ContractError(f"duplicate metadata path: {metadata_path}")
        seen.add(metadata_path)
        if metadata_path.name != ARTIFACT_METADATA_NAME:
            raise ContractError("metadata path must be the exact row-local artifact metadata filename")
        row_root = metadata_path.parent
        if row_root.parent != root:
            raise ContractError("metadata path is not directly below the artifact collection root")
        if _has_symlink_component(row_root) or not row_root.is_dir():
            raise ContractError("metadata row root is missing, symlinked, or not a direct directory")
        if metadata_path.is_symlink():
            raise ContractError("metadata path must not be a symlink")
        result.append((metadata_path, row_root))
    return result


def exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ContractError(f"{label} mismatch")


def require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{label} has missing or unknown fields")
    return value


def regular(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} is missing, symlinked, or not regular: {path}")


def sidecar_hash(path: Path, target: Path, label: str) -> str:
    regular(path, label)
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{label} is not an ASCII hash sidecar") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", content)
    if not match or match.group(2) != target.name or match.group(1) != sha256_file(target):
        raise ContractError(f"{label} does not match {target.name}")
    return sha256_file(path)


def digest(
    value: Any,
    label: str,
    root: Path | None = None,
    *,
    staged_path: Path | None = None,
) -> tuple[Path, str]:
    record = require_keys(value, {"path", "size_bytes", "sha256", "sidecar_path", "sidecar_sha256"}, label)
    declared = Path(record["path"])
    if root is not None:
        if staged_path is None:
            raise ContractError(f"{label} has no exact staged path")
        path = staged_path
        virtual = Path("/output") / path.relative_to(root)
        if declared not in (path, virtual):
            raise ContractError(f"{label} is not the exact row-private staged path")
    else:
        if staged_path is not None:
            raise ContractError(f"{label} has an unexpected staged path")
        path = declared
    regular(path, label)
    if not isinstance(record["size_bytes"], int) or record["size_bytes"] < 1 or record["size_bytes"] > 268435456 or record["size_bytes"] != path.stat().st_size:
        raise ContractError(f"{label} size is stale")
    if not SHA256.fullmatch(record["sha256"]) or record["sha256"] != sha256_file(path):
        raise ContractError(f"{label} content hash is stale")
    declared_sidecar = Path(record["sidecar_path"])
    if root is not None:
        sidecar = path.with_name(path.name + ".sha256")
        virtual_sidecar = Path("/output") / sidecar.relative_to(root)
        if declared_sidecar not in (sidecar, virtual_sidecar):
            raise ContractError(f"{label} sidecar is not the exact row-private staged path")
    else:
        sidecar = declared_sidecar
        if sidecar.parent != path.parent:
            raise ContractError(f"{label} sidecar is not next to its content")
    actual_sidecar_hash = sidecar_hash(sidecar, path, f"{label} sidecar")
    if actual_sidecar_hash != record["sidecar_sha256"]:
        raise ContractError(f"{label} sidecar hash is stale")
    return path, record["sha256"]


def _expected_staged_digest_record(path: Path, row_root: Path) -> dict[str, Any]:
    """Build the exact virtual-container digest association for a staged file."""

    virtual = Path("/output") / path.relative_to(row_root)
    sidecar = path.with_name(path.name + ".sha256")
    return {
        "path": str(virtual),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sidecar_path": str(virtual) + ".sha256",
        "sidecar_sha256": sha256_file(sidecar),
    }


def _validate_exact_row_contents(row_root: Path, target: str) -> None:
    """Reject every row-root entry except the canonical report/build layout."""

    expected_root = {
        "build",
        "hip-runtime-artifact.json",
        "hip-runtime-artifact.json.sha256",
        "report.json",
        "report.json.sha256",
    }
    entries = list(row_root.iterdir())
    if {entry.name for entry in entries} != expected_root:
        raise ContractError("row artifact root has missing, extra, or unexpected entries")
    build_dir = row_root / "build"
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise ContractError("row artifact build entry must be a direct non-symlink directory")
    for name in expected_root - {"build"}:
        regular(row_root / name, f"row artifact {name}")

    expected_build = {
        f"hip-compile-probe-{target}.o",
        f"hip-compile-probe-{target}.o.sha256",
        f"public-runtime-{target}.o",
        f"rmsnorm-kernel-{target}.o",
        f"rmsnorm-api-{target}.o",
        f"public-runtime-{target}.o.sha256",
        f"rmsnorm-kernel-{target}.o.sha256",
        f"rmsnorm-api-{target}.o.sha256",
        f"public-runtime-{target}.elf",
        f"public-runtime-{target}.elf.sha256",
        f"probe-{target}.fatbin",
        f"probe-{target}.fatbin.sha256",
        f"device-code-object-{target}.elf",
        f"device-code-object-{target}.elf.sha256",
    }
    build_entries = list(build_dir.iterdir())
    if {entry.name for entry in build_entries} != expected_build:
        raise ContractError("row artifact build directory has missing, extra, or unexpected entries")
    for entry in build_entries:
        regular(entry, f"row artifact build output {entry.name}")


def check_schema_file(repo: Path, path: str, expected_id: str) -> dict[str, Any]:
    try:
        document = read_json(repo / path)
    except (OSError, RuntimeContractError, ValueError) as exc:
        raise ContractError(f"{path} cannot be read as a checked-in schema") from exc
    if document.get("$schema") != "https://json-schema.org/draft/2020-12/schema" or document.get("$id") != f"https://sllm-project.local/{path}":
        raise ContractError(f"{path} is not a draft-2020-12 versioned schema")
    if document.get("type") != "object" or document.get("additionalProperties") is not False:
        raise ContractError(f"{path} is not closed at the top level")
    if document.get("properties", {}).get("schema_version", {}).get("const") != expected_id:
        raise ContractError(f"{path} has the wrong schema version")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ContractError("jsonschema is required for H3 public-runtime schema validation") from exc
    try:
        Draft202012Validator.check_schema(document)
    except Exception as exc:
        raise ContractError(f"{path} is not a valid JSON schema: {exc}") from exc
    return document


def validate_against_schema(document: Any, schema: dict[str, Any], label: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ContractError("jsonschema is required for H3 public-runtime schema validation") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        raise ContractError(f"{label} fails checked-in schema: {errors[0].message}")


def validate_static(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    compile_schema = check_schema_file(repo, COMPILE_SCHEMA, "hip-runtime-compile-v1")
    check_schema_file(repo, ARTIFACT_SCHEMA, "hip-runtime-artifact-v1")
    check_schema_file(repo, REPORT_SCHEMA, "hip-runtime-public-report-v1")
    check_schema_file(repo, AGGREGATE_SCHEMA, "hip-runtime-aggregate-v1")
    expected_symbols = list(PUBLIC_SYMBOLS)
    symbol_schema = compile_schema.get("properties", {}).get("public_abi_symbols")
    if not isinstance(symbol_schema, dict) or symbol_schema.get("const") != expected_symbols:
        raise ContractError("compile schema public ABI symbols are not the canonical ordered set")
    try:
        matrix_document = read_json(repo / "ci/matrix/hip-runtime-compile-v1.json")
    except (OSError, RuntimeContractError, ValueError) as exc:
        raise ContractError("public-runtime compile matrix cannot be read") from exc
    validate_against_schema(matrix_document, compile_schema, "public-runtime compile matrix")
    toolchain, matrix, rows = validate_matrix(repo)
    if matrix["sources"]["canonical_order"] != list(EXPECTED_SOURCE_PATHS):
        raise ContractError("source canonical order is not exactly the four artifact-compatible files")
    if sorted(matrix["sources"]["canonical_order"]) != list(matrix["sources"]["canonical_order"]):
        raise ContractError("source canonical order is not sorted")
    if matrix["direct_compile_sources"]["canonical_order"] != list(EXPECTED_DIRECT_COMPILE_SOURCE_PATHS):
        raise ContractError("direct compile source canonical order is not exactly the eight audited files")
    if sorted(matrix["direct_compile_sources"]["canonical_order"]) != list(matrix["direct_compile_sources"]["canonical_order"]):
        raise ContractError("direct compile source canonical order is not sorted")
    if matrix["public_abi_symbols"] != sorted(PUBLIC_SYMBOLS):
        raise ContractError("public ABI symbol set is missing, duplicated, or not canonical")
    return toolchain, matrix, rows


def git_identity(repo: Path) -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError(f"git {' '.join(args)} exceeded its bounded inspection") from exc
        if result.returncode != 0:
            raise ContractError(f"git {' '.join(args)} failed")
        if len(result.stdout) + len(result.stderr) > 16 * 1024 * 1024:
            raise ContractError(f"git {' '.join(args)} produced unbounded inspection output")
        return result.stdout.strip()
    commit, tree = run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")
    clean = not run("status", "--porcelain=v1", "--untracked-files=all")
    if not SHA40.fullmatch(commit) or not SHA40.fullmatch(tree):
        raise ContractError("checked-out git identity is not immutable")
    return commit, tree, clean


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


def _render_target_template(template: str, target: str) -> str:
    """Render only the trusted target placeholder in a command template."""

    if not isinstance(template, str):
        raise ContractError("trusted build command template contains a non-string token")
    rendered = template.replace("{target}", target)
    if "{target}" in rendered:
        raise ContractError("trusted build command template has an unresolved target placeholder")
    return rendered


def _placeholder_suffix(template: str, marker: str) -> str:
    """Return the exact suffix for one approved root placeholder."""

    if template.count(marker) != 1 or not template.startswith(marker):
        raise ContractError("trusted build command template has an unsupported root placeholder")
    suffix = template.removeprefix(marker)
    if "{" in suffix or "}" in suffix:
        raise ContractError("trusted build command template has an unresolved placeholder")
    return suffix


def _matches_rendered_command_token(
    actual: Any,
    template: str,
    *,
    target: str,
    repo: Path,
    row_root: Path,
) -> bool:
    """Match one rendered argv token against the closed trusted template set."""

    if not isinstance(actual, str) or "{" in actual or "}" in actual:
        return False
    expected = _render_target_template(template, target)
    has_repo = "{repo}" in expected
    has_build_dir = "{build_dir}" in expected
    if has_repo and has_build_dir:
        raise ContractError("trusted build command template combines repository and build roots")
    if has_repo:
        suffix = _placeholder_suffix(expected, "{repo}")
        return actual in {str(repo.resolve()) + suffix, "/workspace" + suffix}
    if has_build_dir:
        suffix = _placeholder_suffix(expected, "{build_dir}")
        approved_roots = (str(row_root / "build"), "/output/build")
        if actual in {root + suffix for root in approved_roots}:
            return True
        return re.fullmatch(r"/proc/self/fd/[0-9]+" + re.escape(suffix), actual) is not None
    if "{" in expected or "}" in expected:
        raise ContractError("trusted build command template has an unsupported placeholder")
    return actual == expected


def _validate_rendered_build_commands(
    commands: Any,
    *,
    target: str,
    repo: Path,
    row_root: Path,
) -> None:
    """Require the five metadata argv records to match trusted rendered templates."""

    templates = expected_build_commands()
    if not isinstance(commands, list) or len(commands) != len(templates):
        raise ContractError("metadata does not record exactly five pinned argv compiler commands")
    for command_index, (command, template) in enumerate(zip(commands, templates), 1):
        if not isinstance(command, list) or len(command) != len(template):
            raise ContractError(f"metadata command {command_index} has a wrong argv length")
        for token_index, (actual, expected) in enumerate(zip(command, template), 1):
            if not _matches_rendered_command_token(
                actual,
                expected,
                target=target,
                repo=repo,
                row_root=row_root,
            ):
                raise ContractError(
                    f"metadata command {command_index} token {token_index} is not an approved rendered argv token"
                )


def _require_nonnegative_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ContractError(f"{label} is not a finite nonnegative number")
    return float(value)


def _require_nonnegative_integer(value: Any, label: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > limit:
        raise ContractError(f"{label} is not a nonnegative integer within its exact limit")
    return value


def _validate_report_steps(report: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Bind every successful report step to the exact recorded compiler argv."""

    report_started = parse_time(report["started_at"], "report started")
    report_finished = parse_time(report["finished_at"], "report finished")
    if report_started > report_finished:
        raise ContractError("report timestamps are not ordered")
    _require_nonnegative_finite_number(report["duration_seconds"], "report duration")

    build = metadata.get("build")
    if not isinstance(build, dict) or not isinstance(build.get("commands"), list):
        raise ContractError("report metadata has no command list to bind")
    commands = build["commands"]
    steps = report["steps"]
    if not isinstance(steps, list) or len(steps) != 5 or len(commands) != 5:
        raise ContractError("report does not contain exactly five bound compiler steps")

    previous_finished: datetime | None = None
    for index, (step, command) in enumerate(zip(steps, commands), 1):
        step_record = require_keys(
            step,
            {
                "step_id",
                "state",
                "argv",
                "exit_code",
                "started_at",
                "finished_at",
                "duration_seconds",
                "stdout_sha256",
                "stderr_sha256",
                "diagnostic",
                "resource",
            },
            f"report step {index}",
        )
        expected_step_id = f"h3-public-{metadata['target']}.compile-{index}"
        if (
            step_record["step_id"] != expected_step_id
            or step_record["argv"] != command
            or step_record["state"] != "PASS"
            or step_record["exit_code"] != 0
            or step_record["diagnostic"] != ""
        ):
            raise ContractError(f"report step {index} is not the exact successful compiler invocation")
        for digest_name in ("stdout_sha256", "stderr_sha256"):
            digest_value = step_record[digest_name]
            if not isinstance(digest_value, str) or SHA256.fullmatch(digest_value) is None:
                raise ContractError(f"report step {index} {digest_name} is not a canonical SHA-256")

        started = parse_time(step_record["started_at"], f"report step {index} started")
        finished = parse_time(step_record["finished_at"], f"report step {index} finished")
        if started > finished:
            raise ContractError(f"report step {index} timestamps are not ordered")
        if started < report_started or finished > report_finished:
            raise ContractError(f"report step {index} is outside the report timestamp bounds")
        if previous_finished is not None and started < previous_finished:
            raise ContractError(f"report step {index} is not chronologically ordered")
        previous_finished = finished
        _require_nonnegative_finite_number(step_record["duration_seconds"], f"report step {index} duration")

        resource = require_keys(
            step_record["resource"],
            {"output_bytes", "output_limit_bytes", "max_rss_bytes", "max_rss_limit_bytes", "timed_out"},
            f"report step {index} resource",
        )
        if (
            resource["output_limit_bytes"] != EXPECTED_STEP_OUTPUT_LIMIT_BYTES
            or resource["max_rss_limit_bytes"] != EXPECTED_STEP_MAX_RSS_LIMIT_BYTES
            or resource["timed_out"] is not False
        ):
            raise ContractError(f"report step {index} resource limits or timeout state are not exact")
        _require_nonnegative_integer(
            resource["output_bytes"],
            f"report step {index} output bytes",
            EXPECTED_STEP_OUTPUT_LIMIT_BYTES,
        )
        _require_nonnegative_integer(
            resource["max_rss_bytes"],
            f"report step {index} maximum RSS",
            EXPECTED_STEP_MAX_RSS_LIMIT_BYTES,
        )


def validate_metadata(path: Path, repo: Path, *, expected_sha: str | None = None, expected_tree: str | None = None, artifact_root: Path | None = None) -> dict[str, Any]:
    if expected_sha is None or expected_tree is None:
        raise ContractError("strict artifact validation requires expected commit SHA and tree OID")
    if artifact_root is None or artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ContractError("strict artifact validation requires an exact row-private artifact root")
    if path.resolve() != artifact_root.resolve() / "hip-runtime-artifact.json":
        raise ContractError("metadata is not the exact row-private staged metadata path")
    regular(path, "metadata")
    sidecar_hash(path.with_name(path.name + ".sha256"), path, "metadata sidecar")
    metadata = read_json(path)
    required = {"schema_version", "metadata_id", "matrix_row_id", "target", "candidate", "run", "toolchain_id", "matrix_id", "toolchain_manifest_sha256", "matrix_manifest_sha256", "image", "resolved_paths", "source_set", "direct_compile_source_set", "codegen", "build", "host_elf", "device_code_object", "public_abi_symbols", "scope", "execution_environment", "hashes", "timestamps", "duration_seconds"}
    require_keys(metadata, required, "metadata")
    toolchain, matrix, rows = validate_static(repo)
    validate_against_schema(metadata, read_json(repo / ARTIFACT_SCHEMA), "artifact metadata")
    target = metadata["target"]
    row_id = metadata["matrix_row_id"]
    if target not in TARGETS or row_id != f"h3-public-{target}" or metadata["metadata_id"] != f"h3-public-runtime-artifact-{target}" or row_id not in rows:
        raise ContractError("metadata target/row identity is not exact")
    if artifact_root.resolve().name != row_id:
        raise ContractError("artifact root is not the metadata row")
    exact(metadata["toolchain_id"], "rocm-7.14.0", "metadata toolchain")
    exact(metadata["matrix_id"], matrix["matrix_id"], "metadata matrix")
    exact(metadata["toolchain_manifest_sha256"], sha256_json(toolchain), "toolchain manifest hash")
    exact(metadata["matrix_manifest_sha256"], sha256_json(matrix), "matrix manifest hash")
    candidate = require_keys(metadata["candidate"], {"commit_sha", "tree_oid", "reviewed_sha", "tested_sha", "workflow_sha"}, "candidate")
    if not all(isinstance(candidate[key], str) and SHA40.fullmatch(candidate[key]) for key in candidate):
        raise ContractError("metadata candidate identity contains a malformed SHA")
    if len({candidate["commit_sha"], candidate["reviewed_sha"], candidate["tested_sha"], candidate["workflow_sha"]}) != 1:
        raise ContractError("reviewed/tested/workflow/commit SHA values differ")
    if candidate["commit_sha"] != expected_sha:
        raise ContractError("metadata commit is not the expected immutable SHA")
    if candidate["tree_oid"] != expected_tree:
        raise ContractError("metadata tree is not the expected immutable tree OID")
    run = require_keys(metadata["run"], {"run_id", "run_attempt"}, "run")
    if not isinstance(run["run_id"], str) or not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$", run["run_id"]) or not isinstance(run["run_attempt"], int) or run["run_attempt"] < 1:
        raise ContractError("metadata run identity is invalid")
    exact(metadata["image"], {"reference": PINNED_IMAGE, "config_digest": PINNED_CONFIG, "platform": {"os": "linux", "architecture": "amd64"}}, "metadata image")
    exact(metadata["resolved_paths"], {key: toolchain["paths"][key] for key in ("rocm_root", "compiler", "hip_headers", "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy", "llvm_readobj")}, "metadata resolved paths")
    source_set = require_keys(metadata["source_set"], {"canonical_order", "source_set_sha256", "files"}, "source set")
    exact(source_set, matrix["sources"], "metadata source set")
    direct_source_set = require_keys(metadata["direct_compile_source_set"], {"canonical_order", "source_set_sha256", "files"}, "metadata direct compile source set")
    exact(direct_source_set, matrix["direct_compile_sources"], "metadata direct compile source set")
    exact(metadata["codegen"], rows[row_id]["codegen"], "metadata codegen")
    exact(metadata["public_abi_symbols"], sorted(PUBLIC_SYMBOLS), "metadata public symbol set")
    exact(metadata["scope"], EXPECTED_SCOPE, "metadata compile-only scope")
    exact(metadata["execution_environment"], EXPECTED_ENVIRONMENT, "metadata execution environment")
    build = metadata["build"]
    require_keys(build, {"output_directory", "build_directory", "probe_source", "public_runtime_source", "public_runtime_header", "rmsnorm_kernel_source", "rmsnorm_kernel_header", "rmsnorm_api_source", "rmsnorm_api_header", "link_library", "probe_object", "public_runtime_object", "rmsnorm_kernel_object", "rmsnorm_api_object", "host_elf", "probe_fatbin", "device_object", "commands", "generator", "mode", "build_type", "language_standard", "source_tree_output"}, "build")
    if build["source_tree_output"] is not False or build["generator"] != "direct-amdclang++" or build["mode"] != "compile-link" or build["build_type"] != "Release" or build["language_standard"] != "gnu++17":
        raise ContractError("metadata build is not a private direct compile/link")
    for key, relative in (("probe_source", "native/hip/src/hip_compile_probe.hip.cpp"), ("public_runtime_source", "native/hip/src/public_runtime.hip.cpp"), ("public_runtime_header", "native/hip/src/public_runtime_internal.hpp"), ("rmsnorm_kernel_source", "native/hip/src/rmsnorm_kernel.hip.cpp"), ("rmsnorm_kernel_header", "native/hip/src/rmsnorm_kernel_internal.hpp"), ("rmsnorm_api_source", "native/hip/src/rmsnorm_api.cpp"), ("rmsnorm_api_header", "native/hip/src/rmsnorm_api.hpp")):
        declared_source = Path(build[key]).resolve()
        if declared_source not in {(repo / relative).resolve(), Path("/workspace") / relative}:
            raise ContractError(f"metadata {key} is not the approved source")
    build_dir = Path(build["build_directory"])
    output_dir = Path(build["output_directory"])
    row_root = artifact_root.resolve()
    if not output_dir.is_absolute() or output_dir == repo or output_dir.is_relative_to(repo):
        raise ContractError("metadata build output is not row-private and outside the source tree")
    if output_dir not in (row_root, Path("/output")) or build_dir not in (row_root / "build", Path("/output/build")):
        raise ContractError("metadata build output is not the exact row-private staged layout")
    if build["link_library"] != "/opt/rocm/lib/libamdhip64.so":
        raise ContractError("metadata does not record the pinned HIP runtime link library")
    expected_names = {"probe_object": f"hip-compile-probe-{target}.o", "public_runtime_object": f"public-runtime-{target}.o", "rmsnorm_kernel_object": f"rmsnorm-kernel-{target}.o", "rmsnorm_api_object": f"rmsnorm-api-{target}.o", "host_elf": f"public-runtime-{target}.elf", "probe_fatbin": f"probe-{target}.fatbin", "device_object": f"device-code-object-{target}.elf"}
    for key, filename in expected_names.items():
        declared = Path(build[key])
        if declared not in (row_root / "build" / filename, Path("/output/build") / filename):
            raise ContractError(f"metadata {key} is not the exact private target output")
    _validate_rendered_build_commands(build["commands"], target=target, repo=repo, row_root=row_root)
    host = require_keys(metadata["host_elf"], {"format", "machine", "sections", "bundles", "public_symbols", "probe_symbol", "kernel_symbol", "stub_symbols"}, "host ELF evidence")
    exact(host["format"], "ELF64", "host format"); exact(host["machine"], "X86_64", "host machine"); exact(host["bundles"], [f"hipv4-amdgcn-amd-amdhsa--{target}", "host-x86_64-unknown-linux-gnu-"], "host bundles")
    if set(host["sections"]) != {".text", ".hip_fatbin"} or any(not isinstance(item, dict) or item.get("present") is not True or not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 1 for item in host["sections"].values()):
        raise ContractError("host ELF section evidence is incomplete")
    if sorted(item["name"] for item in host["public_symbols"]) != sorted(PUBLIC_SYMBOLS) or any(item.get("defined") is not True for item in host["public_symbols"]) or host["probe_symbol"] != {"name": "sllm_hip_compile_probe", "defined": True} or host["kernel_symbol"] != {"name": "sllm_rmsnorm_baseline_wave32_v1", "defined": True} or host["stub_symbols"]:
        raise ContractError("host ELF public symbols are incomplete or stub-linked")
    device = require_keys(metadata["device_code_object"], {"format", "machine", "target", "ei_abiversion", "e_flags", "code_object_version", "wavefront_size", "features", "sections", "symbols", "source_attribution"}, "device object evidence")
    exact(device["format"], "ELF64", "device format"); exact(device["machine"], "AMDGPU", "device machine"); exact(device["target"], target, "device target"); exact(device["ei_abiversion"], 4, "device ABI"); exact(device["e_flags"], E_FLAGS[target], "device e_flags"); exact(device["code_object_version"], "V6", "device code object version"); exact(device["wavefront_size"], 32, "device wavefront"); exact(device["features"], EXPECTED_FEATURES, "device features"); exact(device["source_attribution"], "hip_compile_probe.hip.cpp", "device source attribution")
    if set(device["sections"]) != {".text"}:
        raise ContractError("device evidence is not probe-only")
    symbols = device["symbols"]
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise ContractError("device evidence is not probe-only")
    probe_symbol = require_keys(symbols[0], {"name", "defined"}, "device probe symbol")
    if probe_symbol["name"] != "sllm_hip_compile_probe" or probe_symbol["defined"] is not True:
        raise ContractError("device evidence is not probe-only")
    hashes = require_keys(metadata["hashes"], EXPECTED_OUTPUT_HASH_KEYS, "output hashes")
    expected_output_paths = {name: row_root / "build" / filename for name, filename in expected_names.items()}
    for name, record in hashes.items():
        digest(record, name, artifact_root, staged_path=expected_output_paths[name])
    for key in ("created_at", "started_at", "finished_at"):
        parse_time(metadata["timestamps"][key], f"metadata {key}")
    if not (parse_time(metadata["timestamps"]["created_at"], "created") <= parse_time(metadata["timestamps"]["started_at"], "started") <= parse_time(metadata["timestamps"]["finished_at"], "finished")):
        raise ContractError("metadata timestamps are not ordered")
    if not isinstance(metadata["duration_seconds"], (int, float)) or metadata["duration_seconds"] < 0:
        raise ContractError("metadata duration is invalid")
    return metadata


def validate_report(path: Path, metadata: dict[str, Any], repo: Path, artifact_root: Path | None = None) -> tuple[dict[str, Any], str, str]:
    if artifact_root is None or artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ContractError("strict report validation requires an exact row-private artifact root")
    if path.resolve() != artifact_root.resolve() / "report.json":
        raise ContractError("report is not the exact row-private staged report path")
    _validate_exact_row_contents(artifact_root, metadata["target"])
    regular(path, "report")
    report = read_json(path)
    required = {"schema_version", "report_id", "row_id", "target", "state", "required", "evidence_mode", "run", "reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "candidate", "toolchain_id", "matrix_id", "matrix_manifest_sha256", "scope", "execution_environment", "compile_only_contract", "steps", "diagnostics", "metadata", "hashes", "started_at", "finished_at", "duration_seconds", "no_output_execution"}
    require_keys(report, required, "report")
    validate_against_schema(report, read_json(repo / REPORT_SCHEMA), "public-runtime report")
    _validate_report_steps(report, metadata)
    if report["state"] != "PASS" or report["required"] is not False or report["evidence_mode"] != "required-ci" or report["row_id"] != metadata["matrix_row_id"] or report["target"] != metadata["target"] or report["candidate"] != metadata["candidate"] or report["reviewed_sha"] != metadata["candidate"]["reviewed_sha"] or report["tested_sha"] != metadata["candidate"]["tested_sha"] or report["workflow_sha"] != metadata["candidate"]["workflow_sha"] or report["git_tree_oid"] != metadata["candidate"]["tree_oid"] or report["run"] != metadata["run"] or report["toolchain_id"] != metadata["toolchain_id"] or report["matrix_id"] != metadata["matrix_id"] or report["matrix_manifest_sha256"] != metadata["matrix_manifest_sha256"] or report["scope"] != EXPECTED_SCOPE or report["execution_environment"] != EXPECTED_ENVIRONMENT or report["compile_only_contract"] != "compile-only; no GPU/support/model/network/fallback evidence" or report["no_output_execution"] is not True or report["diagnostics"]:
        raise ContractError("report does not prove a clean compile-only PASS")
    metadata_file = artifact_root.resolve() / "hip-runtime-artifact.json"
    metadata_sidecar_file = metadata_file.with_name(metadata_file.name + ".sha256")
    metadata_reference = require_keys(report["metadata"], {"path", "sha256", "sidecar_sha256"}, "report metadata association")
    if metadata_reference["path"] != metadata_file.name:
        raise ContractError("report metadata association is not the row-local metadata file")
    if metadata_reference["sha256"] != sha256_file(metadata_file) or metadata_reference["sidecar_sha256"] != sidecar_hash(metadata_sidecar_file, metadata_file, "metadata sidecar"):
        raise ContractError("report metadata hash is stale")
    metadata_hashes = require_keys(metadata["hashes"], EXPECTED_OUTPUT_HASH_KEYS, "metadata hashes")
    report_hashes = require_keys(report["hashes"], EXPECTED_REPORT_HASH_KEYS, "report hashes")
    row_root = artifact_root.resolve()
    expected_output_paths = {
        name: row_root / "build" / filename
        for name, filename in {
            "probe_object": f"hip-compile-probe-{metadata['target']}.o",
            "public_runtime_object": f"public-runtime-{metadata['target']}.o",
            "rmsnorm_kernel_object": f"rmsnorm-kernel-{metadata['target']}.o",
            "rmsnorm_api_object": f"rmsnorm-api-{metadata['target']}.o",
            "host_elf": f"public-runtime-{metadata['target']}.elf",
            "probe_fatbin": f"probe-{metadata['target']}.fatbin",
            "device_object": f"device-code-object-{metadata['target']}.elf",
        }.items()
    }
    metadata_report_record = report_hashes["metadata"]
    expected_metadata_record = _expected_staged_digest_record(metadata_file, row_root)
    if metadata_report_record != expected_metadata_record:
        raise ContractError("report metadata output association is not the exact staged record")
    digest(metadata_report_record, "report metadata output", row_root, staged_path=metadata_file)
    for name, record in metadata_hashes.items():
        report_record = report_hashes[name]
        if report_record != record:
            raise ContractError(f"report output association is not exact: {name}")
        digest(report_record, f"report {name} output", row_root, staged_path=expected_output_paths[name])
    report_sidecar_sha = sidecar_hash(path.with_name(path.name + ".sha256"), path, "report sidecar")
    metadata_path = path.parent / "hip-runtime-artifact.json"
    metadata_record = {"path": str(metadata_path), "size_bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path), "sidecar_path": str(metadata_path) + ".sha256", "sidecar_sha256": sha256_file(Path(str(metadata_path) + ".sha256"))}
    report_record = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "sidecar_path": str(path) + ".sha256", "sidecar_sha256": report_sidecar_sha}
    return report, report_record["sha256"], report_record["sidecar_sha256"]


def validate_contracts(repo: Path, metadata_paths: Iterable[Path], *, expected_sha: str | None = None, expected_tree: str | None = None, artifact_root: Path | None = None) -> None:
    validate_static(repo)
    paths = list(metadata_paths)
    if artifact_root is None:
        if paths:
            raise ContractError("strict artifact validation requires expected identity and artifact root")
        return
    if expected_sha is None or expected_tree is None:
        raise ContractError("strict artifact validation requires expected identity and artifact root")
    if not paths:
        paths = _discover_metadata_paths(artifact_root)
    row_paths = _row_roots_for_collection(paths, artifact_root)
    if not row_paths:
        raise ContractError("artifact validation was requested but the artifact root is empty")
    for metadata_path, row_root in row_paths:
        metadata = validate_metadata(metadata_path, repo, expected_sha=expected_sha, expected_tree=expected_tree, artifact_root=row_root)
        report_path = metadata_path.parent / "report.json"
        if not report_path.exists() or report_path.is_symlink():
            raise ContractError("metadata row is missing its required report.json")
        validate_report(report_path, metadata, repo, row_root)


def validate_h3_manifests(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility-shaped entry point for static contract consumers."""

    toolchain, matrix, _rows = validate_static(repo)
    return toolchain, matrix


def validate_h3_public_runtime_contracts(repo: Path = ROOT, artifact_metadata_paths: Iterable[Path] = (), *, expected_candidate_sha: str | None = None, expected_tree_oid: str | None = None, artifact_root: Path | None = None) -> None:
    validate_contracts(repo, artifact_metadata_paths, expected_sha=expected_candidate_sha, expected_tree=expected_tree_oid, artifact_root=artifact_root)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--artifact-metadata", "--metadata", type=Path, action="append", default=[])
    result.add_argument("--artifact-root", "--artifact-dir", type=Path)
    result.add_argument("--expected-reviewed-sha", "--expected-candidate-sha", "--reviewed-sha", dest="expected_sha")
    result.add_argument("--expected-tree-oid", "--tree-oid", dest="expected_tree")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        metadata_paths = list(args.artifact_metadata)
        if args.artifact_root is not None:
            metadata_paths.extend(_discover_metadata_paths(args.artifact_root))
        validate_contracts(args.repo.resolve(), metadata_paths, expected_sha=args.expected_sha, expected_tree=args.expected_tree, artifact_root=args.artifact_root)
    except (ContractError, RuntimeContractError, OSError, ValueError, KeyError) as exc:
        print(f"H3 public-runtime contract validation: FAIL: {exc}", file=sys.stderr)
        return 1
    print("H3 public-runtime contract validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
