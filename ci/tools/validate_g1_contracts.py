#!/usr/bin/env python3
"""Fail-closed, host-side validators for the model-free G1 runtime contract.

This module never starts the evidence binary and never imports a GPU/runtime
binding.  It validates the checked-in matrix and downloaded evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    exact_sha,
    read_json,
    sha256_file,
    sha256_json,
)

EXPECTED_TARGETS = ("gfx1030", "gfx1201")
EXPECTED_ROWS = tuple(f"g1-{target}" for target in EXPECTED_TARGETS)
EXPECTED_SIZES = (1, 3, 17, 255, 256, 257)
EXPECTED_TOOLCHAIN_ID = "rocm-7.14.0"
TOOLCHAIN_MANIFEST = "ci/toolchains/rocm-7.14.0.json"
MATRIX_MANIFEST = "ci/matrix/g1-runtime-v1.json"
REPORT_SCHEMA = "ci/schema/g1-report-v1.schema.json"
ARTIFACT_SCHEMA = "ci/schema/g1-runtime-artifact-v1.schema.json"
REPORT_NAME = "report.json"
METADATA_NAME = "g1-runtime-artifact.json"
BINARY_NAME = "ullm-hip-evidence"
EXPECTED_BINARY_SUFFIX = ("target", "release", BINARY_NAME)
SHA256_TOKEN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
FORBIDDEN_H3_TEXT = re.compile(
    r"(?:h3|hip[-_]?artifact|device[-_]?code[-_]?object|compile[-_]?only)",
    re.IGNORECASE,
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
    "binary_is_h3_artifact": False,
}
EXPECTED_SCOPE = {
    "model_used": False,
    "cpu_fallback_allowed": False,
    "semantic_op_used": False,
    "byte_exact_verified": True,
    "semantic_numerics_verified": False,
    "required_sizes": list(EXPECTED_SIZES),
}
EXPECTED_COMMAND = {
    "binary": "target/release/ullm-hip-evidence",
    "arguments": ["--timeout-ms", "1000"],
    "documentation": "Dedicated Rust evidence binary; compile artifacts are not executable.",
}
EXPECTED_CODEGEN = {
    "code_object_version": "V6",
    "wavefront_size": 32,
    "features": {
        "xnack": "unsupported",
        "sramecc": "unsupported",
        "generic_processor_version": 0,
    },
}
EXPECTED_E_FLAGS = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}
EXPECTED_BUNDLE_IDS = {
    target: [f"hipv4-amdgcn-amd-amdhsa--{target}", "host-x86_64-unknown-linux-gnu-"]
    for target in EXPECTED_TARGETS
}
EXPECTED_INSPECTOR_TOOLS = {
    "llvm_objcopy": "/opt/rocm/lib/llvm/bin/llvm-objcopy",
    "clang_offload_bundler": "/opt/rocm/lib/llvm/bin/clang-offload-bundler",
    "llvm_readobj": "/opt/rocm/lib/llvm/bin/llvm-readobj",
}
DIAGNOSTIC_KERNEL_TOKEN = "evidence_transform"
HOST_BUNDLE_ID = "host-x86_64-unknown-linux-gnu-"
EXPECTED_LOADER_CONTRACT = {
    "rocm_root": "/opt/rocm",
    "rocm_release": "7.14.0",
    "path": "/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "ld_library_path": "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib",
    "observation_method": "proc-pid-maps-poll-v1",
    "required_libraries": ["libamdhip64.so.7", "libhsa-runtime64.so.1"],
    "inherited_loader_environment": False,
}
EXPECTED_LOADED_LIBRARIES = {
    "libamdhip64.so.7": "/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000",
    "libhsa-runtime64.so.1": "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0",
}
EXPECTED_RUNTIME_BINDING_KEYS = {
    *EXPECTED_LOADER_CONTRACT,
    "loaded_libraries",
}
LIBRARY_NAME_PATTERN = {
    name: re.compile(rf"^{re.escape(name)}(?:\.[0-9][A-Za-z0-9._-]*)?$")
    for name in EXPECTED_LOADER_CONTRACT["required_libraries"]
}
EXPECTED_ROWS_DATA = (
    {
        "row_id": "g1-gfx1030",
        "target": "gfx1030",
        "bdf": "0000:03:00.0",
        "uuid": "GPU-76a08c022586fed6",
        "product": "AMD Radeon Pro V620",
        "timeout_seconds": 300,
        "seed": 1031,
    },
    {
        "row_id": "g1-gfx1201",
        "target": "gfx1201",
        "bdf": "0000:47:00.0",
        "uuid": "GPU-a8e9ddefa2d60f55",
        "product": "AMD Radeon AI PRO R9700",
        "timeout_seconds": 300,
        "seed": 1202,
    },
)

# Inspector output is structured and small.  Keep a generous fixed bound so a
# malformed tool cannot consume host memory while the validator is collecting
# diagnostics.  The builder delegates to the same bounded primitive.
MAX_SUBPROCESS_STDOUT_BYTES = 1024 * 1024
MAX_SUBPROCESS_STDERR_BYTES = 1024 * 1024
MAX_SUBPROCESS_TIMEOUT_SECONDS = 900.0
PROCESS_TERM_GRACE_SECONDS = 1.0
PROCESS_KILL_GRACE_SECONDS = 1.0
SUBPROCESS_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class BoundedCommandResult:
    """The finite output and exit status of one bounded argv-only command."""

    returncode: int
    stdout: bytes
    stderr: bytes


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Permission errors and other uncertainty are unsafe to treat as a
        # clean group, so callers fail closed.
        return True
    return True


def _signal_process_group(process_group_id: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ContractError(f"cannot signal subprocess group {process_group_id}: {exc}") from exc


def _reap_process_group(process: subprocess.Popen[bytes], reason: str) -> None:
    """Bound TERM/KILL/reap cleanup for a started-new-session process."""

    process_group_id = process.pid
    cleanup_error: str | None = None
    try:
        _signal_process_group(process_group_id, signal.SIGTERM)
    except ContractError as exc:
        cleanup_error = str(exc)

    try:
        process.wait(timeout=PROCESS_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            _signal_process_group(process_group_id, signal.SIGKILL)
        except ContractError as exc:
            cleanup_error = cleanup_error or str(exc)
        try:
            process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_error = cleanup_error or "subprocess parent could not be reaped after SIGKILL"

    # A successful parent wait is insufficient: a descendant can retain the
    # process group and its pipe descriptors.  Kill and check the group again,
    # with no unbounded wait or communicate call.
    if _process_group_exists(process_group_id):
        try:
            _signal_process_group(process_group_id, signal.SIGKILL)
        except ContractError as exc:
            cleanup_error = cleanup_error or str(exc)
        if _process_group_exists(process_group_id):
            cleanup_error = cleanup_error or "subprocess process group still exists after SIGKILL"
    if cleanup_error is not None:
        raise ContractError(f"{reason}; bounded subprocess cleanup failed: {cleanup_error}")


def run_bounded_argv(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float,
    max_stdout_bytes: int = MAX_SUBPROCESS_STDOUT_BYTES,
    max_stderr_bytes: int = MAX_SUBPROCESS_STDERR_BYTES,
) -> BoundedCommandResult:
    """Run argv without shell, unbounded pipes, or unbounded final cleanup."""

    command = tuple(str(value) for value in argv)
    if not command or any("\x00" in value for value in command):
        raise ContractError("subprocess argv is empty or contains NUL")
    if timeout <= 0 or timeout > MAX_SUBPROCESS_TIMEOUT_SECONDS:
        raise ContractError(f"subprocess timeout is outside the bounded range: {timeout}")
    if (
        isinstance(max_stdout_bytes, bool)
        or not isinstance(max_stdout_bytes, int)
        or max_stdout_bytes < 1
        or isinstance(max_stderr_bytes, bool)
        or not isinstance(max_stderr_bytes, int)
        or max_stderr_bytes < 1
    ):
        raise ContractError("subprocess output limits must be positive integers")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise ContractError(f"cannot start {' '.join(command)}: {exc}") from exc

    stdout = bytearray()
    stderr = bytearray()
    selector = selectors.DefaultSelector()
    streams = ((process.stdout, stdout, max_stdout_bytes, "stdout"), (process.stderr, stderr, max_stderr_bytes, "stderr"))
    failure: str | None = None
    try:
        for stream, _buffer, _limit, _label in streams:
            if stream is None:
                raise ContractError("subprocess pipe was not created")
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = f"command timed out after {timeout:.3f}s: {' '.join(command)}"
                break
            events = selector.select(remaining)
            if not events:
                failure = f"command timed out after {timeout:.3f}s: {' '.join(command)}"
                break
            for key, _events in events:
                stream, buffer, limit, label = next(item for item in streams if item[0] is key.fileobj)
                try:
                    chunk = os.read(stream.fileno(), min(SUBPROCESS_READ_CHUNK_BYTES, limit - len(buffer) + 1))
                except BlockingIOError:
                    continue
                except OSError as exc:
                    failure = f"cannot read command {label}: {exc}"
                    break
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                if len(buffer) + len(chunk) > limit:
                    failure = f"command {label} output exceeded {limit} bytes: {' '.join(command)}"
                    break
                buffer.extend(chunk)
            if failure is not None:
                break
        if failure is None:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                failure = f"command timed out after {timeout:.3f}s: {' '.join(command)}"
            if failure is None and _process_group_exists(process.pid):
                failure = f"command left descendants in its process group: {' '.join(command)}"
    except (ContractError, OSError) as exc:
        failure = str(exc)
    finally:
        try:
            selector.close()
        finally:
            for stream, _buffer, _limit, _label in streams:
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    if failure is not None:
        _reap_process_group(process, failure)
        raise ContractError(failure)
    return BoundedCommandResult(process.returncode, bytes(stdout), bytes(stderr))


def _schema_validator(schema: dict[str, Any], label: str) -> Any:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - locked host dependency
        raise ContractError("jsonschema is required for G1 contract validation") from exc
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_schema(document: Any, schema: dict[str, Any], label: str) -> None:
    errors = sorted(_schema_validator(schema, label).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ContractError(f"{label} schema validation failed: {detail}")


def _matrix_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/g1_runtime_matrix",
    }


def load_contract(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    for relative in (REPORT_SCHEMA, ARTIFACT_SCHEMA, MATRIX_MANIFEST):
        _require_regular(repo / relative, f"G1 contract file {relative}")
    report_schema = read_json(repo / REPORT_SCHEMA)
    artifact_schema = read_json(repo / ARTIFACT_SCHEMA)
    matrix = read_json(repo / MATRIX_MANIFEST)
    if not all(isinstance(value, dict) for value in (report_schema, artifact_schema, matrix)):
        raise ContractError("G1 schema and matrix documents must be JSON objects")
    _schema_validator(artifact_schema, "G1 runtime artifact schema")
    validate_schema(matrix, _matrix_schema(report_schema), "G1 runtime matrix")
    rows = {row["row_id"]: row for row in matrix.get("rows", []) if isinstance(row, dict) and "row_id" in row}
    return report_schema, matrix, rows


def row_by_id(matrix: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ContractError("G1 matrix rows are missing")
    for row in rows:
        if isinstance(row, dict) and row.get("row_id") == row_id:
            return row
    raise ContractError(f"unknown G1 row: {row_id}")


def validate_g1_matrix(repo: Path = ROOT) -> dict[str, Any]:
    """Require the closed, ordered canonical two-row G1 matrix."""

    report_schema, matrix, _rows = load_contract(repo)
    expected_top = {
        "$schema", "schema_version", "matrix_id", "revision", "tier", "required",
        "toolchain_id", "execution", "scope", "command", "rows",
    }
    if set(matrix) != expected_top:
        raise ContractError("G1 matrix has unknown or missing top-level keys")
    if matrix["$schema"] != "https://ullm-project.local/ci/schema/g1-report-v1.schema.json#/$defs/g1_runtime_matrix":
        raise ContractError("G1 matrix schema binding drifted")
    if matrix["schema_version"] != "g1-runtime-v1" or matrix["matrix_id"] != "g1-runtime-v1" or matrix["revision"] != 1:
        raise ContractError("G1 matrix identity/revision drifted")
    if matrix["tier"] != "tier_g1" or matrix["required"] is not True:
        raise ContractError("G1 matrix must be a required tier_g1 contract")
    if matrix["toolchain_id"] != EXPECTED_TOOLCHAIN_ID:
        raise ContractError("G1 matrix toolchain drifted")
    if matrix["execution"] != EXPECTED_EXECUTION:
        raise ContractError("G1 execution lock/security contract drifted")
    if matrix["scope"] != EXPECTED_SCOPE:
        raise ContractError("G1 must remain model-free and non-semantic")
    if matrix["command"] != EXPECTED_COMMAND:
        raise ContractError("G1 must use the dedicated release evidence binary")
    if matrix["rows"] != list(EXPECTED_ROWS_DATA):
        raise ContractError("G1 rows must be exactly the ordered canonical gfx1030/gfx1201 pair")
    if len({row["bdf"] for row in matrix["rows"]}) != 2 or len({row["uuid"] for row in matrix["rows"]}) != 2:
        raise ContractError("G1 canonical GPU identities must be distinct")
    for row in matrix["rows"]:
        if any(FORBIDDEN_H3_TEXT.search(str(value)) for value in row.values()):
            raise ContractError(f"G1 row contains an H3 name or path: {row['row_id']}")
    # The schema is checked after the closed semantic checks so an accidental
    # schema relaxation cannot make this matrix open-ended.
    validate_schema(matrix, _matrix_schema(report_schema), "G1 runtime matrix")
    return matrix


def _require_regular(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing, symlinked, or not a regular file")


def _sidecar_sha256(sidecar: Path, target: Path, label: str) -> str:
    _require_regular(target, f"{label} target")
    _require_regular(sidecar, label)
    try:
        text = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{label} is not an ASCII sidecar") from exc
    expected = f"{sha256_file(target)}  {target.name}\n"
    if text != expected:
        raise ContractError(f"{label} is stale, malformed, or names the wrong file")
    return sha256_file(sidecar)


def _tail_is_dedicated_binary(path_value: Any, label: str) -> None:
    if not isinstance(path_value, str) or "\x00" in path_value:
        raise ContractError(f"{label} is not a valid absolute path")
    path = Path(path_value)
    if not path.is_absolute() or path.parts[-3:] != EXPECTED_BINARY_SUFFIX or "." in path.parts or ".." in path.parts:
        raise ContractError(f"{label} is not a target/release/{BINARY_NAME} path")
    if FORBIDDEN_H3_TEXT.search(path_value):
        raise ContractError(f"{label} contains an H3 path")


def _path_outside_repo(path_value: str, repo: Path, label: str) -> None:
    """Reject source paths that point back into the checked-out source tree."""

    path = Path(path_value)
    try:
        path.resolve(strict=False).relative_to(repo.resolve())
    except ValueError:
        return
    raise ContractError(f"{label} must be outside the source tree")


def _validate_source_artifact_path(path_value: Any, repo: Path, label: str) -> None:
    _tail_is_dedicated_binary(path_value, label)
    _path_outside_repo(path_value, repo, label)
    path = Path(path_value)
    try:
        relative = path.resolve(strict=False).relative_to(Path("/tmp"))
    except ValueError as exc:
        raise ContractError(f"{label} must be below /tmp") from exc
    if not relative.parts or not relative.parts[0].startswith("ullm-g1-"):
        raise ContractError(f"{label} must be below a private ullm-g1-* staging root")
    root = Path("/tmp") / relative.parts[0]
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"{label} private staging root is missing or unsafe")
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ContractError(f"{label} private staging root is not owned privately by the current user")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ContractError(f"{label} traverses a symlink")


def _manifest_hashes(repo: Path) -> dict[str, str]:
    for relative in (TOOLCHAIN_MANIFEST, MATRIX_MANIFEST, REPORT_SCHEMA, ARTIFACT_SCHEMA):
        path = repo / relative
        _require_regular(path, f"G1 contract manifest {relative}")
    matrix = read_json(repo / MATRIX_MANIFEST)
    if not isinstance(matrix, dict):
        raise ContractError("G1 matrix manifest must be an object")
    return {
        "toolchain_manifest_sha256": sha256_file(repo / TOOLCHAIN_MANIFEST),
        "matrix_manifest_sha256": sha256_json(matrix),
        "report_schema_sha256": sha256_file(repo / REPORT_SCHEMA),
        "artifact_schema_sha256": sha256_file(repo / ARTIFACT_SCHEMA),
    }


def _parse_report_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z", value
    ):
        raise ContractError(f"{label} is not a strict UTC RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{label} has no timezone")
    return parsed.astimezone(timezone.utc)


def _validate_observation_device(observation: Mapping[str, Any], row: Mapping[str, Any], label: str) -> datetime:
    if observation["available"] is not True or observation["reliable"] is not True:
        raise ContractError(f"{label} health/process evidence is unavailable or unreliable")
    if observation["device"] != {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]}:
        raise ContractError(f"{label} evidence is not bound to the canonical GPU")
    return _parse_report_time(observation["observed_at"], f"{label}.observed_at")


def _expected_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "run_id": identity.get("run_id"),
        "run_attempt": identity.get("run_attempt"),
        "reviewed_sha": identity.get("reviewed_sha"),
        "tested_sha": identity.get("tested_sha"),
        "workflow_sha": identity.get("workflow_sha"),
        "git_tree_oid": identity.get("git_tree_oid"),
    }
    if not isinstance(result["run_id"], str) or not RUN_ID_TOKEN.fullmatch(result["run_id"]):
        raise ContractError("G1 run_id is invalid")
    if isinstance(result["run_attempt"], bool) or not isinstance(result["run_attempt"], int) or result["run_attempt"] < 1:
        raise ContractError("G1 run_attempt is invalid")
    for name in ("reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid"):
        exact_sha(result[name], name)
    if len({result["reviewed_sha"], result["tested_sha"], result["workflow_sha"]}) != 1:
        raise ContractError("G1 reviewed/tested/workflow SHA values differ")
    return result


def _expected_row(expected: Mapping[str, Any]) -> dict[str, Any]:
    if set(expected) != set(EXPECTED_ROWS_DATA[0]):
        raise ContractError("G1 expected row has unknown or missing keys")
    if expected not in EXPECTED_ROWS_DATA:
        raise ContractError("G1 expected row is not canonical")
    return dict(expected)


def _inspection_tool_result(
    argv: Sequence[str],
    *,
    tool_runner: Callable[..., Any] | None,
) -> tuple[int, bytes, bytes]:
    """Run one pinned inspector without allowing a shell or tool fallback."""

    command = tuple(str(value) for value in argv)
    if not command or any("\x00" in value for value in command):
        raise ContractError("G1 inspector command is empty or contains NUL")
    try:
        if tool_runner is None:
            result = run_bounded_argv(
                command,
                timeout=60.0,
                max_stdout_bytes=MAX_SUBPROCESS_STDOUT_BYTES,
                max_stderr_bytes=MAX_SUBPROCESS_STDERR_BYTES,
            )
        else:
            result = tool_runner(command, timeout=60.0)
    except (ContractError, OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ContractError(f"G1 inspector command failed to start: {command[0]}") from exc
    try:
        stdout = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout).encode("utf-8")
        stderr = result.stderr if isinstance(result.stderr, bytes) else str(result.stderr).encode("utf-8")
        if len(stdout) > MAX_SUBPROCESS_STDOUT_BYTES or len(stderr) > MAX_SUBPROCESS_STDERR_BYTES:
            raise ContractError(f"G1 inspector output exceeded the bounded limit: {command[0]}")
        return int(result.returncode), stdout, stderr
    except (AttributeError, ContractError, TypeError, ValueError) as exc:
        raise ContractError(f"G1 inspector returned an invalid result: {command[0]}") from exc


def _inspection_text(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"{label} returned non-UTF-8 output") from exc


def _readobj_sections(output: str) -> tuple[dict[str, int], list[str]]:
    sections: dict[str, int] = {}
    names: list[str] = []
    blocks = re.findall(r"(?ms)^\s*Section \{\n(.*?)^\s*\}", output)
    for block in blocks:
        name = re.search(r"(?m)^\s*Name: ([^\n]+)", block)
        size = re.search(r"(?m)^\s*Size: (0x[0-9a-fA-F]+|[0-9]+)", block)
        if name is None or size is None:
            raise ContractError("llvm-readobj section output is malformed")
        section_name = re.sub(r"\s+\(\d+\)$", "", name.group(1).strip())
        names.append(section_name)
        sections[section_name] = int(size.group(1), 0)
    return sections, names


def _readobj_symbols(output: str) -> list[dict[str, str]]:
    symbols: list[dict[str, str]] = []
    blocks = re.findall(r"(?ms)^\s*Symbol \{\n(.*?)^\s*\}", output)
    for block in blocks:
        name = re.search(r"(?m)^\s*Name: ([^\n]+)", block)
        section = re.search(r"(?m)^\s*Section: ([^\n]+)", block)
        symbol_type = re.search(r"(?m)^\s*Type: ([^\n]+)", block)
        if name is None or section is None or symbol_type is None:
            raise ContractError("llvm-readobj symbol output is malformed")
        symbols.append(
            {
                "name": re.sub(r"\s+\(\d+\)$", "", name.group(1).strip()),
                "section": re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", section.group(1).strip()),
                "type": re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", symbol_type.group(1).strip()),
            }
        )
    if not symbols:
        raise ContractError("device code object has no symbol table entries")
    return symbols


def parse_g1_host_readobj(output: str) -> dict[str, Any]:
    """Parse pinned llvm-readobj host output and prove one fatbin section."""

    formats = re.findall(r"(?m)^Format: ([^\n]+)$", output)
    architectures = re.findall(r"(?m)^Arch: ([^\n]+)$", output)
    if formats != ["elf64-x86-64"] or architectures != ["x86_64"]:
        raise ContractError("final G1 executable is not one ELF64 x86-64 host object")
    sections, section_names = _readobj_sections(output)
    if section_names.count(".hip_fatbin") != 1 or sections.get(".hip_fatbin", 0) <= 0:
        raise ContractError("final G1 executable does not contain exactly one non-empty .hip_fatbin")
    return {
        "format": "ELF64",
        "machine": "X86_64",
        "hip_fatbin_size_bytes": sections[".hip_fatbin"],
    }


def parse_g1_device_readobj(output: str, target: str) -> dict[str, Any]:
    """Parse one extracted AMDGPU code object using the existing H3 tuple."""

    if target not in EXPECTED_TARGETS:
        raise ContractError(f"unknown G1 device target: {target}")
    formats = re.findall(r"(?m)^Format: ([^\n]+)$", output)
    architectures = re.findall(r"(?m)^Arch: ([^\n]+)$", output)
    if formats != ["elf64-amdgpu"] or architectures != ["amdgcn"]:
        raise ContractError("extracted G1 device object is not ELF64 AMDGPU")
    abi = re.findall(r"(?m)^\s*ABIVersion: (\d+)$", output)
    header_output = output.split("Sections [", 1)[0]
    flags = re.findall(r"(?m)^\s*Flags \[ \(0x([0-9a-fA-F]+)\)$", header_output)
    observed_flags = [f"0x{int(value, 16):08x}" for value in flags]
    if abi != ["4"] or observed_flags != [EXPECTED_E_FLAGS[target]]:
        raise ContractError("device code object ABI/e_flags do not prove the pinned Code Object V6 target")

    target_values = re.findall(r"(?m)^amdhsa\.target:\s*([^\s]+)\s*$", output)
    expected_target = f"amdgcn-amd-amdhsa--{target}"
    if target_values != [expected_target]:
        raise ContractError("device code object target is missing, duplicated, suffixed, or unexpected")
    wavefront_values = re.findall(r"(?m)^\s*\.wavefront_size:\s*(\d+)\s*$", output)
    if wavefront_values != ["32"]:
        raise ContractError("device code object does not prove exactly one wave32 kernel")

    # The canonical H3 target spelling has no xnack/sramecc suffix.  If a
    # future llvm-readobj version prints these fields explicitly, require the
    # same canonical unsupported values instead of silently ignoring them.
    for feature in ("xnack", "sramecc"):
        values = re.findall(rf"(?im)^\s*{feature}\s*[:=]\s*([^\s]+)\s*$", output)
        if values and values != ["unsupported"]:
            raise ContractError(f"device code object {feature} feature is not unsupported")
    generic_values = re.findall(r"(?im)^\s*generic_processor_version\s*[:=]\s*(\d+)\s*$", output)
    if generic_values and generic_values != ["0"]:
        raise ContractError("device code object generic_processor_version is not zero")

    sections, section_names = _readobj_sections(output)
    if section_names.count(".text") != 1 or sections.get(".text", 0) <= 0:
        raise ContractError("device code object does not contain exactly one non-empty .text")
    symbols = _readobj_symbols(output)
    function_symbols = [
        symbol["name"]
        for symbol in symbols
        if symbol["type"] == "Function" and symbol["section"] == ".text"
    ]
    kernel_names = re.findall(r"(?m)^\s*\.name:\s*([^\s]+)\s*$", output)
    kernel_symbols = re.findall(r"(?m)^\s*\.symbol:\s*([^\s]+)\s*$", output)
    if len(kernel_names) != 1 or len(kernel_symbols) != 1:
        raise ContractError("device code object has missing or multiple kernel metadata entries")
    kernel_name = kernel_names[0]
    if (
        DIAGNOSTIC_KERNEL_TOKEN not in kernel_name
        or function_symbols != [kernel_name]
        or kernel_symbols != [f"{kernel_name}.kd"]
    ):
        raise ContractError("device code object has no defined diagnostic kernel symbol")
    return {
        "target": target,
        "code_object_version": "V6",
        "ei_abiversion": 4,
        "e_flags": EXPECTED_E_FLAGS[target],
        "wavefront_size": 32,
        "features": dict(EXPECTED_CODEGEN["features"]),
        "diagnostic_kernel_symbol": kernel_name,
    }


def _artifact_snapshot(path: Path, label: str) -> tuple[int, str]:
    """Capture the regular file's size and content digest for later binding."""

    _require_regular(path, label)
    try:
        return path.stat().st_size, sha256_file(path)
    except OSError as exc:
        raise ContractError(f"cannot snapshot {label}: {exc}") from exc


def _assert_artifact_snapshot(path: Path, expected: tuple[int, str], label: str) -> None:
    actual = _artifact_snapshot(path, label)
    if actual != expected:
        raise ContractError(
            f"{label} changed during inspection: "
            f"expected size/sha256={expected[0]}/{expected[1]}, "
            f"observed={actual[0]}/{actual[1]}"
        )


def _validate_inspection_output_location(path: Path, temporary_path: Path, label: str) -> None:
    """Keep an inspector output path below the private TemporaryDirectory."""

    try:
        if temporary_path.is_symlink() or not temporary_path.is_dir():
            raise ContractError("private inspection directory is missing or unsafe")
        temporary_root = temporary_path.resolve(strict=True)
        temporary_stat = temporary_root.stat()
        candidate = path.resolve(strict=False)
        candidate.relative_to(temporary_root)
        path.parent.resolve(strict=True).relative_to(temporary_root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} is outside the private inspection directory") from exc
    if temporary_stat.st_uid != os.getuid() or temporary_stat.st_mode & 0o077:
        raise ContractError("private inspection directory is not owned privately by the current user")
    if candidate == temporary_root or path.is_symlink():
        raise ContractError(f"{label} is an unsafe inspection output path")


def _prepare_inspection_output(path: Path, temporary_path: Path, label: str) -> None:
    _validate_inspection_output_location(path, temporary_path, label)
    if path.exists() or path.is_symlink():
        raise ContractError(f"{label} already exists or is symlinked")


def _require_inspection_output(path: Path, temporary_path: Path, label: str) -> None:
    _validate_inspection_output_location(path, temporary_path, label)
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing, symlinked, or not a regular file")
    try:
        if path.stat().st_size <= 0:
            raise ContractError(f"{label} is empty")
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {exc}") from exc


def inspect_g1_runtime_artifact(
    artifact_path: Path,
    target: str,
    *,
    tool_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Inspect the final staged executable through pinned ROCm tools."""

    if target not in EXPECTED_TARGETS:
        raise ContractError(f"unknown G1 artifact target: {target}")
    _require_regular(artifact_path, "G1 final staged executable")
    if not os.access(artifact_path, os.X_OK):
        raise ContractError("G1 final staged executable is not executable")
    artifact_snapshot = _artifact_snapshot(artifact_path, "G1 final staged executable")
    try:
        tools = {name: Path(path) for name, path in EXPECTED_INSPECTOR_TOOLS.items()}
        for name, path in tools.items():
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(Path("/opt/rocm"))
            except (OSError, ValueError):
                raise ContractError(f"pinned G1 inspector is missing or outside /opt/rocm: {name}")
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ContractError(f"pinned G1 inspector is missing or not executable: {name}")

        host_code, host_stdout, host_err = _inspection_tool_result(
            [str(tools["llvm_readobj"]), "--file-headers", "--sections", str(artifact_path)],
            tool_runner=tool_runner,
        )
        if host_code != 0:
            raise ContractError(f"llvm-readobj failed for final G1 executable: {_inspection_text(host_err, 'llvm-readobj stderr').strip()}")
        host_output = _inspection_text(host_stdout, "llvm-readobj host output")
        host = parse_g1_host_readobj(host_output)

        with tempfile.TemporaryDirectory(prefix="ullm-g1-inspect-", dir="/tmp") as temporary:
            temporary_path = Path(temporary)
            fatbin = temporary_path / "embedded.hip_fatbin"
            objcopy_output = temporary_path / "objcopy-output"
            device = temporary_path / "device-code-object.elf"
            _prepare_inspection_output(fatbin, temporary_path, "G1 .hip_fatbin output")
            _prepare_inspection_output(objcopy_output, temporary_path, "G1 llvm-objcopy output artifact")
            _prepare_inspection_output(device, temporary_path, "G1 device code-object output")
            if objcopy_output.resolve(strict=False) == artifact_path.resolve(strict=True):
                raise ContractError("G1 llvm-objcopy output artifact aliases the inspected executable")
            dump_code, _dump_out, dump_err = _inspection_tool_result(
                [
                    str(tools["llvm_objcopy"]),
                    f"--dump-section=.hip_fatbin={fatbin}",
                    str(artifact_path),
                    str(objcopy_output),
                ],
                tool_runner=tool_runner,
            )
            if dump_code != 0:
                raise ContractError(f"cannot extract a non-empty .hip_fatbin: {_inspection_text(dump_err, 'llvm-objcopy stderr').strip()}")
            _require_inspection_output(fatbin, temporary_path, "G1 .hip_fatbin output")
            if objcopy_output.exists() or objcopy_output.is_symlink():
                _require_inspection_output(objcopy_output, temporary_path, "G1 llvm-objcopy output artifact")
            list_code, list_out, list_err = _inspection_tool_result(
                [str(tools["clang_offload_bundler"]), "--list", "--type=o", f"--input={fatbin}"],
                tool_runner=tool_runner,
            )
            if list_code != 0:
                raise ContractError(f"clang-offload-bundler list failed: {_inspection_text(list_err, 'bundler stderr').strip()}")
            bundle_ids = [line.strip() for line in _inspection_text(list_out, "bundler list output").splitlines() if line.strip()]
            expected_bundle_ids = [f"hipv4-amdgcn-amd-amdhsa--{target}", HOST_BUNDLE_ID]
            if bundle_ids != expected_bundle_ids:
                raise ContractError(f"G1 embedded bundle list is not exactly one target plus host: {bundle_ids}")
            unbundle_code, _unbundle_out, unbundle_err = _inspection_tool_result(
                [
                    str(tools["clang_offload_bundler"]),
                    "--unbundle",
                    "--type=o",
                    f"--targets={expected_bundle_ids[0]}",
                    f"--input={fatbin}",
                    f"--output={device}",
                ],
                tool_runner=tool_runner,
            )
            if unbundle_code != 0:
                raise ContractError(f"device code object extraction failed: {_inspection_text(unbundle_err, 'bundler stderr').strip()}")
            _require_inspection_output(device, temporary_path, "G1 device code-object output")
            device_code, device_stdout, device_err = _inspection_tool_result(
                [
                    str(tools["llvm_readobj"]),
                    "--file-headers",
                    "--sections",
                    "--symbols",
                    "--notes",
                    str(device),
                ],
                tool_runner=tool_runner,
            )
            if device_code != 0:
                raise ContractError(f"llvm-readobj failed for device code object: {_inspection_text(device_err, 'llvm-readobj stderr').strip()}")
            device_output = _inspection_text(device_stdout, "llvm-readobj device output")
            observed_device = parse_g1_device_readobj(device_output, target)
            device_digest = sha256_file(device)

        return {
            "observed": {
                **observed_device,
                "bundles": [
                    {"id": expected_bundle_ids[0], "target": target},
                    {"id": HOST_BUNDLE_ID, "target": "host"},
                ],
            },
            "device_code_sha256": device_digest,
            "host_bundle": host,
        }
    finally:
        _assert_artifact_snapshot(artifact_path, artifact_snapshot, "G1 final staged executable")


def validate_artifact_metadata(
    metadata: dict[str, Any],
    artifact_path: Path | None = None,
    metadata_path: Path | None = None,
    expected: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
    repo: Path = ROOT,
    *,
    tool_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate the dedicated G1 binary metadata, independently of H3."""

    artifact_schema = read_json(repo / ARTIFACT_SCHEMA)
    if not isinstance(artifact_schema, dict):
        raise ContractError("G1 runtime artifact schema must be an object")
    validate_schema(metadata, artifact_schema, "G1 runtime artifact metadata")
    target = metadata.get("target")
    if target not in EXPECTED_TARGETS:
        raise ContractError("G1 artifact metadata target is unknown")
    row = _expected_row(expected or next(row for row in EXPECTED_ROWS_DATA if row["target"] == target))
    if metadata["metadata_id"] != f"g1-runtime-artifact-{target}" or metadata["row_id"] != row["row_id"]:
        raise ContractError("G1 artifact metadata id/row mismatch")
    if metadata["target"] != target or metadata["gpu"] != {"bdf": row["bdf"], "uuid": row["uuid"], "target": target}:
        raise ContractError("G1 artifact metadata GPU identity mismatch")
    if metadata["toolchain_id"] != EXPECTED_TOOLCHAIN_ID:
        raise ContractError("G1 artifact metadata toolchain mismatch")
    candidate = metadata["candidate"]
    candidate_shas = [exact_sha(candidate[name], name) for name in ("reviewed_sha", "tested_sha", "workflow_sha")]
    exact_sha(candidate["git_tree_oid"], "git_tree_oid")
    if len(set(candidate_shas)) != 1 or candidate["worktree_clean"] is not True or candidate["revision_input"] != "full-sha":
        raise ContractError("G1 artifact metadata candidate is not one clean immutable full-SHA identity")
    if identity is not None:
        expected_identity = _expected_identity(identity)
        if metadata["candidate"] != {
            "reviewed_sha": expected_identity["reviewed_sha"],
            "tested_sha": expected_identity["tested_sha"],
            "workflow_sha": expected_identity["workflow_sha"],
            "git_tree_oid": expected_identity["git_tree_oid"],
            "worktree_clean": True,
            "revision_input": "full-sha",
        }:
            raise ContractError("G1 artifact metadata candidate identity is stale")
    manifest_hashes = _manifest_hashes(repo)
    if metadata["toolchain_manifest_sha256"] != manifest_hashes["toolchain_manifest_sha256"]:
        raise ContractError("G1 artifact metadata toolchain manifest hash is stale")
    if metadata["matrix_manifest_sha256"] != manifest_hashes["matrix_manifest_sha256"]:
        raise ContractError("G1 artifact metadata matrix manifest hash is stale")
    if metadata["artifact_schema_sha256"] != manifest_hashes["artifact_schema_sha256"]:
        raise ContractError("G1 artifact metadata schema hash is stale")
    expected_observed = {
        "target": target,
        "code_object_version": "V6",
        "ei_abiversion": 4,
        "e_flags": EXPECTED_E_FLAGS[target],
        "wavefront_size": 32,
        "features": dict(EXPECTED_CODEGEN["features"]),
        "bundles": [
            {"id": f"hipv4-amdgcn-amd-amdhsa--{target}", "target": target},
            {"id": HOST_BUNDLE_ID, "target": "host"},
        ],
    }
    observed_metadata = metadata["observed"]
    if any(observed_metadata[name] != value for name, value in expected_observed.items()):
        raise ContractError("G1 artifact metadata observed tuple or bundle list is not canonical")
    if DIAGNOSTIC_KERNEL_TOKEN not in observed_metadata["diagnostic_kernel_symbol"]:
        raise ContractError("G1 artifact metadata has no diagnostic kernel symbol")
    scope = metadata["scope"]
    if scope != {
        "model_used": False,
        "cpu_fallback_allowed": False,
        "cpu_fallback_used": False,
        "binary_command": ["target/release/ullm-hip-evidence", "--timeout-ms", "1000"],
    }:
        raise ContractError("G1 artifact metadata scope is not dedicated runtime scope")
    artifact = metadata["artifact"]
    _validate_source_artifact_path(artifact["path"], repo, "G1 metadata artifact path")
    source_artifact = Path(artifact["path"])
    _require_regular(source_artifact, "G1 source runtime binary")
    source_sidecar_sha = _sidecar_sha256(
        source_artifact.with_name(source_artifact.name + ".sha256"),
        source_artifact,
        "G1 source artifact sidecar",
    )
    source_record = {
        "path": artifact["path"],
        "size_bytes": source_artifact.stat().st_size,
        "sha256": sha256_file(source_artifact),
        "sidecar_sha256": source_sidecar_sha,
        "kind": "dedicated-rust-evidence-binary",
    }
    if artifact != source_record:
        raise ContractError("G1 source artifact content or sidecar hash is stale")
    if artifact_path is not None:
        _require_regular(artifact_path, "G1 dedicated runtime binary")
        if artifact_path.name != BINARY_NAME:
            raise ContractError("G1 dedicated runtime binary has the wrong name")
        actual_size = artifact_path.stat().st_size
        actual_sha = sha256_file(artifact_path)
        actual_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
        actual_sidecar_sha = _sidecar_sha256(actual_sidecar, artifact_path, "G1 artifact sidecar")
        if {
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "sidecar_sha256": actual_sidecar_sha,
        } != {
            "size_bytes": source_record["size_bytes"],
            "sha256": source_record["sha256"],
            "sidecar_sha256": source_record["sidecar_sha256"],
        }:
            raise ContractError("G1 staged binary does not match the source artifact")
    if metadata_path is not None:
        _require_regular(metadata_path, "G1 runtime metadata")
        _sidecar_sha256(metadata_path.with_name(metadata_path.name + ".sha256"), metadata_path, "G1 metadata sidecar")
    if artifact_path is not None:
        observed = inspect_g1_runtime_artifact(artifact_path, target, tool_runner=tool_runner)
        if metadata["observed"] != observed["observed"]:
            raise ContractError("G1 metadata observed code-object tuple or bundle list is stale")
        if metadata["device_code_sha256"] != observed["device_code_sha256"]:
            raise ContractError("G1 metadata device-code digest is stale")
    return {
        "row_id": row["row_id"],
        "target": target,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "toolchain_id": metadata["toolchain_id"],
        "artifact_sha256": artifact["sha256"],
        "artifact_sidecar_sha256": artifact["sidecar_sha256"],
    }


validate_runtime_artifact_metadata = validate_artifact_metadata


def validate_runtime_binding(binding: Any) -> None:
    """Require the exact loader binding emitted by the G1 runner."""

    if not isinstance(binding, dict) or set(binding) != EXPECTED_RUNTIME_BINDING_KEYS:
        raise ContractError("G1 runtime_binding has unknown or missing keys")
    for key, expected in EXPECTED_LOADER_CONTRACT.items():
        if binding[key] != expected:
            raise ContractError(f"G1 runtime_binding {key} is not canonical")
    loaded = binding["loaded_libraries"]
    required = EXPECTED_LOADER_CONTRACT["required_libraries"]
    if not isinstance(loaded, dict) or set(loaded) != set(required):
        raise ContractError("G1 runtime_binding loaded library set is incomplete or over-broad")
    seen_paths: set[str] = set()
    for name in required:
        path = loaded[name]
        if not isinstance(path, str) or not Path(path).is_absolute() or not path.startswith("/opt/rocm/"):
            raise ContractError(f"G1 runtime_binding path for {name} is not an absolute ROCm path")
        if not LIBRARY_NAME_PATTERN[name].fullmatch(Path(path).name):
            raise ContractError(f"G1 runtime_binding path for {name} has the wrong soname")
        if path != EXPECTED_LOADED_LIBRARIES[name]:
            raise ContractError(f"G1 runtime_binding path for {name} is not canonical")
        if path in seen_paths:
            raise ContractError(f"G1 runtime_binding has duplicate loaded path for {name}")
        seen_paths.add(path)


def validate_report(
    report: dict[str, Any],
    expected: Mapping[str, Any],
    identity: Mapping[str, Any],
    artifact_path: Path,
    metadata_path: Path,
    matrix: Mapping[str, Any] | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Require a complete PASS report with six exact byte-exact cases."""

    report_schema = read_json(repo / "ci/schema/g1-report-v1.schema.json")
    if not isinstance(report_schema, dict):
        raise ContractError("G1 report schema must be an object")
    validate_schema(report, report_schema, "G1 report")
    row = _expected_row(expected)
    expected_identity = _expected_identity(identity)
    if report["row_id"] != row["row_id"] or report["target"] != row["target"]:
        raise ContractError("G1 report row/target mismatch")
    if report["report_id"] != f"{row['row_id']}.{expected_identity['run_id']}.{expected_identity['run_attempt']}":
        raise ContractError("G1 report id is stale or mismatched")
    if report["state"] != "PASS" or report["required"] is not True or report["error"] is not None:
        raise ContractError("G1 report is not a PASS")
    if report["run_id"] != expected_identity["run_id"] or report["run_attempt"] != expected_identity["run_attempt"]:
        raise ContractError("G1 report run identity mismatch")
    candidate = report["candidate"]
    if candidate != {
        "reviewed_sha": expected_identity["reviewed_sha"],
        "tested_sha": expected_identity["tested_sha"],
        "workflow_sha": expected_identity["workflow_sha"],
        "git_tree_oid": expected_identity["git_tree_oid"],
        "worktree_clean": True,
        "revision_input": "full-sha",
    }:
        raise ContractError("G1 report candidate identity is stale or unclean")
    if report["device"] != {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]}:
        raise ContractError("G1 report device identity is not canonical")
    validate_runtime_binding(report["runtime_binding"])
    created_at = _parse_report_time(report["created_at"], "G1 report created_at")
    started_at = _parse_report_time(report["started_at"], "G1 report started_at")
    finished_at = _parse_report_time(report["finished_at"], "G1 report finished_at")
    now = datetime.now(timezone.utc)
    if not created_at <= started_at <= finished_at or finished_at > now:
        raise ContractError("G1 report timestamps are stale, unordered, or in the future")
    elapsed = (finished_at - started_at).total_seconds()
    if abs(report["duration_seconds"] - elapsed) > 0.01:
        raise ContractError("G1 report duration does not match its timestamps")
    if abs(report["execution"]["duration_seconds"] - report["duration_seconds"]) > 0.01:
        raise ContractError("G1 execution duration does not match report duration")
    pre_health_at = _validate_observation_device(report["health_pre"], row, "G1 health_pre")
    post_health_at = _validate_observation_device(report["health_post"], row, "G1 health_post")
    pre_process_at = _validate_observation_device(report["process_pre"], row, "G1 process_pre")
    post_process_at = _validate_observation_device(report["process_post"], row, "G1 process_post")
    if report["health_pre"]["state"] != "OK" or report["health_post"]["state"] != "OK":
        raise ContractError("G1 report health evidence is not healthy before and after execution")
    if report["process_pre"]["state"] != "CLEAN" or report["process_post"]["state"] != "CLEAN":
        raise ContractError("G1 report process evidence is not clean before and after execution")
    if not pre_health_at <= started_at or not pre_process_at <= started_at:
        raise ContractError("G1 pre-execution evidence was observed after execution started")
    if not post_health_at >= finished_at or not post_process_at >= finished_at:
        raise ContractError("G1 post-execution evidence was observed before execution finished")
    if any(observed_at > now for observed_at in (pre_health_at, post_health_at, pre_process_at, post_process_at)):
        raise ContractError("G1 health/process evidence contains a future observation")
    if not math.isfinite(float(report["health_pre"]["temperature_c"])) or not math.isfinite(float(report["health_post"]["temperature_c"])):
        raise ContractError("G1 health evidence temperature is not finite")
    if report["health_pre"]["device"] != report["device"] or report["health_post"]["device"] != report["device"]:
        raise ContractError("G1 health evidence device binding is stale")
    if report["process_pre"]["device"] != report["device"] or report["process_post"]["device"] != report["device"]:
        raise ContractError("G1 process evidence device binding is stale")
    command = ["target/release/ullm-hip-evidence", "--timeout-ms", "1000"]
    execution = report["execution"]
    if execution["command"] != command or execution["command_sha256"] != sha256_json(command):
        raise ContractError("G1 report command is not the dedicated release binary")
    if execution["exit_code"] != 0 or execution["timed_out"] is not False or execution["crashed"] is not False:
        raise ContractError("G1 report execution did not finish cleanly")
    if report["scope"] != {
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
    }:
        raise ContractError("G1 report totals or scope are not the exact model-free runtime contract")
    sizes = [case["size"] for case in report["cases"]]
    if sizes != list(EXPECTED_SIZES):
        raise ContractError("G1 report cases are not the exact ordered boundary sizes")
    for case in report["cases"]:
        if case != {
            "size": case["size"],
            "state": "PASS",
            "byte_exact": True,
            "allocation_count": 2,
            "copy_count": 2,
            "kernel_dispatch_count": 1,
            "dispatch_count": 1,
            "timed_out": False,
            "fallback_used": False,
        }:
            raise ContractError("G1 report contains a non-byte-exact, fallback, or non-dispatched case")
    if {
        "allocation_count": sum(case["allocation_count"] for case in report["cases"]),
        "copy_count": sum(case["copy_count"] for case in report["cases"]),
        "kernel_dispatch_count": sum(case["kernel_dispatch_count"] for case in report["cases"]),
        "dispatch_count": sum(case["dispatch_count"] for case in report["cases"]),
    } != {
        "allocation_count": report["scope"]["allocation_count"],
        "copy_count": report["scope"]["copy_count"],
        "kernel_dispatch_count": report["scope"]["kernel_dispatch_count"],
        "dispatch_count": report["scope"]["dispatch_count"],
    }:
        raise ContractError("G1 report totals do not equal the six case totals")
    if report["execution"]["stdout_sha256"] == "" or report["execution"]["stderr_sha256"] == "":
        raise ContractError("G1 report output hashes are missing")
    _require_regular(artifact_path, "G1 dedicated runtime binary")
    _require_regular(metadata_path, "G1 runtime metadata")
    metadata_sidecar = metadata_path.with_name(metadata_path.name + ".sha256")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    metadata_sidecar_sha = _sidecar_sha256(metadata_sidecar, metadata_path, "G1 metadata sidecar")
    artifact_sidecar_sha = _sidecar_sha256(artifact_sidecar, artifact_path, "G1 artifact sidecar")
    artifact_record = report["artifact"]
    _validate_source_artifact_path(artifact_record["artifact_path"], repo, "G1 report artifact path")
    if not isinstance(artifact_record["metadata_path"], str) or not Path(artifact_record["metadata_path"]).is_absolute() or Path(artifact_record["metadata_path"]).name != METADATA_NAME:
        raise ContractError("G1 report metadata path is not the dedicated metadata file")
    if artifact_record["metadata_path"] != str(metadata_path.resolve()):
        raise ContractError("G1 report metadata path is not the staged metadata file")
    if artifact_record["staged_artifact_path"] != str(artifact_path.resolve()):
        raise ContractError("G1 report staged artifact path is not the downloaded row artifact")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict) or artifact_record["artifact_path"] != metadata["artifact"]["path"]:
        raise ContractError("G1 report source artifact path is inconsistent with metadata")
    manifest_hashes = _manifest_hashes(repo)
    if artifact_record != {
        "metadata_path": artifact_record["metadata_path"],
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_sidecar_sha256": metadata_sidecar_sha,
        "artifact_path": artifact_record["artifact_path"],
        "staged_artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_sidecar_sha256": artifact_sidecar_sha,
        "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
        "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
        "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
        "target": row["target"],
        "row_id": row["row_id"],
        "h3_executable_used": False,
    }:
        raise ContractError("G1 report artifact hashes or binding are stale")
    return {
        "row_id": row["row_id"],
        "target": row["target"],
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "state": "PASS",
        "run_id": report["run_id"],
        "run_attempt": report["run_attempt"],
        "reviewed_sha": candidate["reviewed_sha"],
        "tested_sha": candidate["tested_sha"],
        "workflow_sha": candidate["workflow_sha"],
        "git_tree_oid": candidate["git_tree_oid"],
        "toolchain_id": EXPECTED_TOOLCHAIN_ID,
        "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
        "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
        "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
        "artifact_path": artifact_record["artifact_path"],
        "staged_artifact_path": artifact_record["staged_artifact_path"],
        "artifact_sha256": artifact_record["artifact_sha256"],
        "metadata_sha256": artifact_record["metadata_sha256"],
        "report_sha256": sha256_json(report),
    }


def validate_row(
    row_dir: Path,
    row_id: str,
    expected: Mapping[str, Any],
    identity: Mapping[str, Any],
    matrix: Mapping[str, Any] | None = None,
    repo: Path = ROOT,
    *,
    tool_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate one downloaded G1 row and every content sidecar."""

    if matrix is not None and matrix != validate_g1_matrix(repo):
        raise ContractError("G1 row was validated against a stale or non-canonical matrix")
    if not row_dir.is_absolute() or row_dir.name != row_id or row_id not in EXPECTED_ROWS or row_dir.is_symlink() or not row_dir.is_dir():
        raise ContractError(f"{row_id}: G1 row directory is missing or unsafe")
    artifact_path = row_dir / BINARY_NAME
    metadata_path = row_dir / METADATA_NAME
    expected_files = {
        REPORT_NAME, REPORT_NAME + ".sha256",
        METADATA_NAME, METADATA_NAME + ".sha256",
        BINARY_NAME, BINARY_NAME + ".sha256",
    }
    if {path.name for path in row_dir.iterdir()} != expected_files:
        raise ContractError(f"{row_id}: G1 row has missing, duplicate, or unknown files")
    for path in row_dir.iterdir():
        _require_regular(path, f"{row_id}: {path.name}")
    report_path = row_dir / REPORT_NAME
    _sidecar_sha256(report_path.with_name(report_path.name + ".sha256"), report_path, f"{row_id}: report sidecar")
    _sidecar_sha256(metadata_path.with_name(metadata_path.name + ".sha256"), metadata_path, f"{row_id}: metadata sidecar")
    _sidecar_sha256(artifact_path.with_name(artifact_path.name + ".sha256"), artifact_path, f"{row_id}: artifact sidecar")
    metadata = read_json(metadata_path)
    report = read_json(report_path)
    if not isinstance(metadata, dict) or not isinstance(report, dict):
        raise ContractError(f"{row_id}: G1 report/metadata must be JSON objects")
    metadata_summary = validate_artifact_metadata(
        metadata, artifact_path, metadata_path, expected, identity, repo,
        tool_runner=tool_runner,
    )
    report_summary = validate_report(
        report, expected, identity, artifact_path, metadata_path, matrix, repo
    )
    if report_summary["artifact_sha256"] != metadata_summary["artifact_sha256"]:
        raise ContractError(f"{row_id}: report and metadata bind different binaries")
    return {
        **report_summary,
        "report_sha256": sha256_file(report_path),
        "artifact_sidecar_sha256": metadata_summary["artifact_sidecar_sha256"],
        "metadata_sidecar_sha256": sha256_file(metadata_path.with_name(metadata_path.name + ".sha256")),
        "report_sidecar_sha256": sha256_file(report_path.with_name(report_path.name + ".sha256")),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-only", action="store_true")
    result.add_argument("--repo", type=Path, default=ROOT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_g1_matrix(args.repo.resolve())
    except (ContractError, OSError, TypeError, ValueError) as exc:
        print(f"G1 contracts: FAIL: {exc}", file=sys.stderr)
        return 1
    print("H0 G1 static contracts: PASS matrix=2 rows sizes=1,3,17,255,256,257; no GPU evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
