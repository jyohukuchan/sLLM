#!/usr/bin/env python3
"""Host-only contract and negative-path tests for H3 public-runtime evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from run_h3_public_runtime_compile import (  # noqa: E402
    E_FLAGS,
    EXPECTED_FEATURES,
    EXPECTED_DIRECT_COMPILE_SOURCE_PATHS,
    EXPECTED_SOURCE_PATHS,
    EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS,
    CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS,
    KERNEL_SYMBOLS,
    PUBLIC_SYMBOLS,
    RuntimeContractError,
    TARGETS,
    declared_public_symbols,
    render_commands,
    require_clean_checkout,
    main as run_h3_public_runtime_main,
)
from validate_h3_public_runtime_contracts import (  # noqa: E402
    ContractError,
    EXPECTED_ENVIRONMENT,
    EXPECTED_SCOPE,
    read_json,
    validate_against_schema,
    validate_metadata,
    validate_report,
    validate_contracts,
    validate_static,
    main as validate_main,
)
from common import ContractError as ManifestContractError  # noqa: E402
from validate_json_manifests import validate_h3_public_runtime_workflow  # noqa: E402
import validate_matrix as matrix_registry  # noqa: E402


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def _is_rust_ident_start(character: str) -> bool:
    return character == "_" or character.isalpha()


def _is_rust_ident_continue(character: str) -> bool:
    return character == "_" or character.isalnum()


def _skip_rust_whitespace(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _rust_identifier_end(source: str, index: int) -> int:
    if index >= len(source) or not _is_rust_ident_start(source[index]):
        raise AssertionError("expected Rust identifier")
    index += 1
    while index < len(source) and _is_rust_ident_continue(source[index]):
        index += 1
    return index


def _skip_rust_escape(source: str, index: int, *, byte_literal: bool) -> int:
    """Skip one Rust escape and reject malformed escape sequences."""

    if index + 1 >= len(source):
        raise AssertionError("unterminated Rust escape sequence")
    escaped = source[index + 1]
    if escaped in "nrt0\\'\"":
        return index + 2
    if escaped in "\r\n":
        if escaped == "\r" and index + 2 < len(source) and source[index + 2] == "\n":
            return index + 3
        return index + 2
    if escaped == "x":
        end = index + 4
        digits = source[index + 2 : end]
        if len(digits) != 2 or any(digit not in "0123456789abcdefABCDEF" for digit in digits):
            raise AssertionError("malformed Rust hexadecimal escape")
        if byte_literal and int(digits, 16) > 0x7F:
            raise AssertionError("non-ASCII Rust byte escape")
        return end
    if escaped == "u" and not byte_literal:
        cursor = index + 2
        if cursor >= len(source) or source[cursor] != "{":
            raise AssertionError("malformed Rust Unicode escape")
        cursor += 1
        digit_start = cursor
        while cursor < len(source) and source[cursor] in "0123456789abcdefABCDEF":
            cursor += 1
        digits = source[digit_start:cursor]
        if not 1 <= len(digits) <= 6 or cursor >= len(source) or source[cursor] != "}":
            raise AssertionError("malformed Rust Unicode escape")
        value = int(digits, 16)
        if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
            raise AssertionError("invalid Rust Unicode scalar escape")
        return cursor + 1
    raise AssertionError(f"malformed Rust escape sequence: \\{escaped}")


def _skip_rust_string(source: str, index: int, *, byte_literal: bool) -> tuple[int, str]:
    """Skip an ordinary Rust string and return its end plus unescaped source body."""

    start = index + 1
    index += 1
    while index < len(source):
        character = source[index]
        if character == "\\":
            index = _skip_rust_escape(source, index, byte_literal=byte_literal)
            continue
        if character == '"':
            return index + 1, source[start:index]
        if character in "\r\n":
            raise AssertionError("unterminated Rust string literal")
        index += 1
    raise AssertionError("unterminated Rust string literal")


def _skip_rust_char(source: str, index: int, *, byte_literal: bool) -> int:
    """Skip and minimally validate an ordinary or byte character literal."""

    index += 1
    if index >= len(source):
        raise AssertionError("unterminated Rust character literal")
    if source[index] == "\\":
        index = _skip_rust_escape(source, index, byte_literal=byte_literal)
    else:
        character = source[index]
        if character in "\r\n'" or (byte_literal and ord(character) > 0x7F):
            raise AssertionError("malformed Rust character literal")
        index += 1
    if index >= len(source) or source[index] != "'":
        raise AssertionError("unterminated Rust character literal")
    return index + 1


def _raw_string_end(source: str, index: int, *, byte_literal: bool) -> int | None:
    """Return the end of a Rust raw string prefix, or None when it is not one."""

    prefix = "br" if byte_literal else "r"
    if not source.startswith(prefix, index):
        return None
    cursor = index + len(prefix)
    hash_count = 0
    while cursor < len(source) and source[cursor] == "#":
        hash_count += 1
        cursor += 1
    if cursor >= len(source) or source[cursor] != '"':
        return None
    closing = '"' + ("#" * hash_count)
    end = source.find(closing, cursor + 1)
    if end < 0:
        raise AssertionError("unterminated Rust raw string literal")
    return end + len(closing)


def _skip_rust_block_comment(source: str, index: int) -> int:
    depth = 1
    index += 2
    while index < len(source) and depth:
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
        else:
            index += 1
    if depth:
        raise AssertionError("unterminated Rust block comment")
    return index


_RUST_DELIMITERS = {"(": ")", "[": "]", "{": "}"}
_RUST_CLOSERS = set(_RUST_DELIMITERS.values())


def _skip_rust_delimited(source: str, index: int) -> int:
    """Skip a balanced token tree while still validating nested literals/comments."""

    opening = source[index]
    if opening not in _RUST_DELIMITERS:
        raise AssertionError("expected Rust delimiter")
    stack = [_RUST_DELIMITERS[opening]]
    index += 1
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            index = _skip_rust_block_comment(source, index)
            continue
        raw_end = _raw_string_end(source, index, byte_literal=False)
        if raw_end is not None:
            index = raw_end
            continue
        byte_raw_end = _raw_string_end(source, index, byte_literal=True)
        if byte_raw_end is not None:
            index = byte_raw_end
            continue
        if source[index] == '"':
            index, _ = _skip_rust_string(source, index, byte_literal=False)
            continue
        if source.startswith('b"', index):
            index, _ = _skip_rust_string(source, index + 1, byte_literal=True)
            continue
        if source[index] == "'":
            if index + 1 < len(source) and _is_rust_ident_start(source[index + 1]):
                end = _rust_identifier_end(source, index + 1)
                if end < len(source) and source[end] == "'":
                    index = _skip_rust_char(source, index, byte_literal=False)
                else:
                    index = end
            else:
                index = _skip_rust_char(source, index, byte_literal=False)
            continue
        if source.startswith("b'", index):
            index = _skip_rust_char(source, index + 1, byte_literal=True)
            continue
        character = source[index]
        if character in _RUST_DELIMITERS:
            stack.append(_RUST_DELIMITERS[character])
            index += 1
            continue
        if character in _RUST_CLOSERS:
            if character != stack[-1]:
                raise AssertionError("mismatched Rust delimiters")
            stack.pop()
            index += 1
            if not stack:
                return index
            continue
        index += 1
    raise AssertionError("unterminated Rust delimited token tree")


def _attribute_start(source: str, index: int) -> int | None:
    if source.startswith("#![", index):
        return index + 2
    if source.startswith("#[", index):
        return index + 1
    return None


def _macro_invocation_end(source: str, index: int) -> int | None:
    """Find non-println macro bodies so their token trees cannot become registrations."""

    if index >= len(source) or not _is_rust_ident_start(source[index]):
        return None
    cursor = _rust_identifier_end(source, index)
    final_name = source[index:cursor]
    cursor = _skip_rust_whitespace(source, cursor)
    while source.startswith("::", cursor):
        cursor = _skip_rust_whitespace(source, cursor + 2)
        if cursor >= len(source) or not _is_rust_ident_start(source[cursor]):
            return None
        name_start = cursor
        cursor = _rust_identifier_end(source, cursor)
        final_name = source[name_start:cursor]
        cursor = _skip_rust_whitespace(source, cursor)
    if final_name in {"if", "while"}:
        return None
    if cursor >= len(source) or source[cursor] != "!":
        return None
    cursor = _skip_rust_whitespace(source, cursor + 1)
    if final_name == "macro_rules":
        if cursor >= len(source) or not _is_rust_ident_start(source[cursor]):
            raise AssertionError("malformed Rust macro_rules declaration")
        cursor = _rust_identifier_end(source, cursor)
        cursor = _skip_rust_whitespace(source, cursor)
    if final_name == "println":
        return None
    if cursor >= len(source) or source[cursor] not in _RUST_DELIMITERS:
        raise AssertionError("malformed Rust macro invocation")
    return _skip_rust_delimited(source, cursor)


def _rust_tokens(source: str) -> list[tuple[str, str]]:
    """Tokenize enough Rust syntax to inspect build-script path expressions."""

    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            index = _skip_rust_block_comment(source, index)
            continue
        attribute_end = _attribute_start(source, index)
        if attribute_end is not None:
            index = _skip_rust_delimited(source, attribute_end)
            continue
        macro_end = _macro_invocation_end(source, index)
        if macro_end is not None:
            index = macro_end
            continue
        raw_end = _raw_string_end(source, index, byte_literal=False)
        if raw_end is not None:
            index = raw_end
            continue
        byte_raw_end = _raw_string_end(source, index, byte_literal=True)
        if byte_raw_end is not None:
            index = byte_raw_end
            continue
        if character == '"':
            end, value = _skip_rust_string(source, index, byte_literal=False)
            tokens.append(("string", value))
            index = end
            continue
        if source.startswith('b"', index):
            index, _ = _skip_rust_string(source, index + 1, byte_literal=True)
            continue
        if source.startswith("b'", index):
            index = _skip_rust_char(source, index + 1, byte_literal=True)
            continue
        if character == "'":
            if index + 1 < len(source) and _is_rust_ident_start(source[index + 1]):
                end = _rust_identifier_end(source, index + 1)
                if end < len(source) and source[end] == "'":
                    index = _skip_rust_char(source, index, byte_literal=False)
                else:
                    tokens.append(("lifetime", source[index:end]))
                    index = end
            else:
                index = _skip_rust_char(source, index, byte_literal=False)
            continue
        if _is_rust_ident_start(character):
            start = index
            index = _rust_identifier_end(source, index)
            tokens.append(("ident", source[start:index]))
            continue
        tokens.append(("punct", character))
        index += 1
    return tokens


def _build_script_rerun_paths(build_script: Path) -> list[tuple[str, str]]:
    """Read actual rerun registrations from the build script's path expressions."""

    tokens = _rust_tokens(build_script.read_text(encoding="utf-8"))
    known_paths = {"manifest_dir": "crates/sllm-hip-sys"}
    path_bindings: dict[str, str] = {}
    registrations: list[tuple[str, str]] = []

    for index, token in enumerate(tokens):
        if (
            token == ("ident", "let")
            and index + 9 < len(tokens)
            and tokens[index + 1][0] == "ident"
            and tokens[index + 2] == ("punct", "=")
            and tokens[index + 3][0] == "ident"
            and tokens[index + 4] == ("punct", ".")
            and tokens[index + 5] == ("ident", "join")
            and tokens[index + 6] == ("punct", "(")
            and tokens[index + 7][0] == "string"
            and tokens[index + 8] == ("punct", ")")
            and tokens[index + 9] == ("punct", ";")
        ):
            name = tokens[index + 1][1]
            base = tokens[index + 3][1]
            if base not in known_paths:
                continue
            path = posixpath.normpath(posixpath.join(known_paths[base], tokens[index + 7][1]))
            if path == "." or path.startswith("../"):
                raise AssertionError(f"build-script path escapes the repository: {path}")
            if name in known_paths:
                raise AssertionError(f"duplicate build-script path binding: {name}")
            known_paths[name] = path
            path_bindings[name] = path

        if (
            token == ("ident", "println")
            and index + 11 < len(tokens)
            and tokens[index + 1] == ("punct", "!")
            and tokens[index + 2] == ("punct", "(")
            and tokens[index + 3] == ("string", "cargo:rerun-if-changed={}")
            and tokens[index + 4] == ("punct", ",")
            and tokens[index + 5][0] == "ident"
            and tokens[index + 6] == ("punct", ".")
            and tokens[index + 7] == ("ident", "display")
            and tokens[index + 8] == ("punct", "(")
            and tokens[index + 9] == ("punct", ")")
            and tokens[index + 10] == ("punct", ")")
            and tokens[index + 11] == ("punct", ";")
        ):
            name = tokens[index + 5][1]
            if name not in path_bindings:
                raise AssertionError(f"rerun registration uses an unknown path binding: {name}")
            registrations.append((name, path_bindings[name]))

    return registrations


def _assert_h3_build_inputs_registered(build_script: Path) -> None:
    required = set(EXPECTED_DIRECT_COMPILE_SOURCE_PATHS) | {"native/hip/CMakeLists.txt"}
    registrations = _build_script_rerun_paths(build_script)
    paths = [path for _name, path in registrations]
    counts = {path: paths.count(path) for path in required}
    missing = sorted(path for path, count in counts.items() if count == 0)
    duplicate = sorted(path for path, count in counts.items() if count > 1)
    if missing or duplicate:
        raise AssertionError(
            f"H3 public-runtime rerun registration is not exact: missing={missing}, duplicate={duplicate}"
        )
    for relative_path in required:
        path = ROOT / relative_path
        if path.is_symlink() or not path.is_file():
            raise AssertionError(f"H3 public-runtime input is not a regular file: {relative_path}")


class ArtifactFixture:
    """Create a small metadata/artifact tree without compiling or executing it."""

    def __init__(self, target: str = "gfx1030") -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="sllm-h3-public-runtime-")
        self.row_dir = Path(self.tempdir.name) / f"h3-public-{target}"
        self.build_dir = self.row_dir / "build"
        self.build_dir.mkdir(parents=True)
        self.target = target
        self.row_id = f"h3-public-{target}"
        self.matrix = json.loads((ROOT / "ci/matrix/hip-runtime-compile-v1.json").read_text())
        self.toolchain = json.loads((ROOT / "ci/toolchains/rocm-7.14.0.json").read_text())
        self.row = next(item for item in self.matrix["rows"] if item["row_id"] == self.row_id)
        self.identity = {key: "a" * 40 for key in ("commit_sha", "reviewed_sha", "tested_sha", "workflow_sha")}
        self.identity["tree_oid"] = "b" * 40
        self.run = {"run_id": "unit-h3-public-runtime", "run_attempt": 1}
        self.output_paths = {
            "probe_object": self.build_dir / f"hip-compile-probe-{target}.o",
            "public_runtime_object": self.build_dir / f"public-runtime-{target}.o",
            "rmsnorm_kernel_object": self.build_dir / f"rmsnorm-kernel-{target}.o",
            "rmsnorm_api_object": self.build_dir / f"rmsnorm-api-{target}.o",
            "host_elf": self.build_dir / f"public-runtime-{target}.elf",
            "probe_fatbin": self.build_dir / f"probe-{target}.fatbin",
            "device_object": self.build_dir / f"device-code-object-{target}.elf",
        }
        for name, path in self.output_paths.items():
            path.write_bytes((name + "-fake-output\n").encode())
            write_sidecar(path)
        self.metadata_path = self.row_dir / "hip-runtime-artifact.json"
        self.report_path = self.row_dir / "report.json"
        self.metadata = self._make_metadata()
        self.write_metadata()
        self.report = self._make_report()
        self.write_report()

    def _make_metadata(self) -> dict[str, object]:
        hashes: dict[str, dict[str, object]] = {}
        for name, path in self.output_paths.items():
            sidecar = path.with_name(path.name + ".sha256")
            staged_path = Path("/output/build") / path.name
            hashes[name] = {
                "path": str(staged_path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sidecar_path": str(staged_path) + ".sha256",
                "sidecar_sha256": sha256_file(sidecar),
            }
        build_commands = render_commands(self.row, ROOT, Path("/proc/self/fd/5"))
        timestamp = {
            "created_at": "2026-08-04T00:00:00Z",
            "started_at": "2026-08-04T00:00:01Z",
            "finished_at": "2026-08-04T00:00:02Z",
        }
        return {
            "schema_version": "hip-runtime-artifact-v1",
            "metadata_id": f"h3-public-runtime-artifact-{self.target}",
            "matrix_row_id": self.row_id,
            "target": self.target,
            "candidate": copy.deepcopy(self.identity),
            "run": copy.deepcopy(self.run),
            "toolchain_id": "rocm-7.14.0",
            "matrix_id": self.matrix["matrix_id"],
            "toolchain_manifest_sha256": hashlib.sha256(canonical_bytes(self.toolchain)).hexdigest(),
            "matrix_manifest_sha256": hashlib.sha256(canonical_bytes(self.matrix)).hexdigest(),
            "image": {
                "reference": "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7",
                "config_digest": "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf",
                "platform": {"os": "linux", "architecture": "amd64"},
            },
            "resolved_paths": {
                key: self.toolchain["paths"][key]
                for key in ("rocm_root", "compiler", "hip_headers", "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy", "llvm_readobj")
            },
            "source_set": copy.deepcopy(self.matrix["sources"]),
            "direct_compile_source_set": copy.deepcopy(self.matrix["direct_compile_sources"]),
            "codegen": copy.deepcopy(self.row["codegen"]),
            "build": {
                "output_directory": "/output",
                "build_directory": "/output/build",
                "probe_source": str(ROOT / EXPECTED_SOURCE_PATHS[1]),
                "public_runtime_source": str(ROOT / EXPECTED_SOURCE_PATHS[2]),
                "public_runtime_header": str(ROOT / EXPECTED_SOURCE_PATHS[3]),
                "rmsnorm_kernel_source": str(ROOT / "native/hip/src/rmsnorm_kernel.hip.cpp"),
                "rmsnorm_kernel_header": str(ROOT / "native/hip/src/rmsnorm_kernel_internal.hpp"),
                "rmsnorm_api_source": str(ROOT / "native/hip/src/rmsnorm_api.cpp"),
                "rmsnorm_api_header": str(ROOT / "native/hip/src/rmsnorm_api.hpp"),
                "link_library": "/opt/rocm/lib/libamdhip64.so",
                **{key: str(Path("/output/build") / path.name) for key, path in self.output_paths.items()},
                "commands": build_commands,
                "generator": "direct-amdclang++",
                "mode": "compile-link",
                "build_type": "Release",
                "language_standard": "gnu++17",
                "source_tree_output": False,
            },
            "host_elf": {
                "format": "ELF64",
                "machine": "X86_64",
                "sections": {".text": {"present": True, "size_bytes": 1}, ".hip_fatbin": {"present": True, "size_bytes": 1}},
                "bundles": [f"hipv4-amdgcn-amd-amdhsa--{self.target}", "host-x86_64-unknown-linux-gnu-"],
                "public_symbols": [{"name": name, "defined": True} for name in sorted(PUBLIC_SYMBOLS)],
                "probe_symbol": {"name": "sllm_hip_compile_probe", "defined": True},
                "kernel_symbol": {"name": "sllm_rmsnorm_baseline_wave32_v1", "defined": True},
                "stub_symbols": [],
            },
            "device_code_object": {
                "format": "ELF64",
                "machine": "AMDGPU",
                "target": self.target,
                "ei_abiversion": 4,
                "e_flags": E_FLAGS[self.target],
                "code_object_version": "V6",
                "wavefront_size": 32,
                "features": copy.deepcopy(EXPECTED_FEATURES),
                "sections": {".text": {"present": True, "size_bytes": 1}},
                "symbols": [{"name": "sllm_hip_compile_probe", "defined": True}],
                "source_attribution": "hip_compile_probe.hip.cpp",
            },
            "public_abi_symbols": sorted(PUBLIC_SYMBOLS),
            "scope": copy.deepcopy(EXPECTED_SCOPE),
            "execution_environment": copy.deepcopy(EXPECTED_ENVIRONMENT),
            "hashes": hashes,
            "timestamps": timestamp,
            "duration_seconds": 1,
        }

    def _make_report(self) -> dict[str, object]:
        metadata_path = self.metadata_path
        metadata_hash = {
            "path": "/output/hip-runtime-artifact.json",
            "size_bytes": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
            "sidecar_path": "/output/hip-runtime-artifact.json.sha256",
            "sidecar_sha256": sha256_file(metadata_path.with_name(metadata_path.name + ".sha256")),
        }
        empty_output_digest = hashlib.sha256(b"").hexdigest()
        steps = []
        for index, command in enumerate(self.metadata["build"]["commands"], 1):
            started_millis = (index - 1) * 100
            finished_millis = index * 100
            steps.append(
                {
                    "step_id": f"{self.row_id}.compile-{index}",
                    "state": "PASS",
                    "argv": copy.deepcopy(command),
                    "exit_code": 0,
                    "started_at": f"2026-08-04T00:00:01.{started_millis:03d}Z",
                    "finished_at": f"2026-08-04T00:00:01.{finished_millis:03d}Z",
                    "duration_seconds": 0.1,
                    "stdout_sha256": empty_output_digest,
                    "stderr_sha256": empty_output_digest,
                    "diagnostic": "",
                    "resource": {
                        "output_bytes": 0,
                        "output_limit_bytes": 16777216,
                        "max_rss_bytes": 1024,
                        "max_rss_limit_bytes": 4294967296,
                        "timed_out": False,
                    },
                }
            )
        return {
            "schema_version": "hip-runtime-public-report-v1",
            "report_id": f"h3-public-runtime.{self.target}.unit-h3-public-runtime.1",
            "row_id": self.row_id,
            "target": self.target,
            "state": "PASS",
            "required": False,
            "evidence_mode": "required-ci",
            "run": copy.deepcopy(self.run),
            "reviewed_sha": self.identity["reviewed_sha"],
            "tested_sha": self.identity["tested_sha"],
            "workflow_sha": self.identity["workflow_sha"],
            "git_tree_oid": self.identity["tree_oid"],
            "candidate": copy.deepcopy(self.identity),
            "toolchain_id": "rocm-7.14.0",
            "matrix_id": self.matrix["matrix_id"],
            "matrix_manifest_sha256": hashlib.sha256(canonical_bytes(self.matrix)).hexdigest(),
            "scope": copy.deepcopy(EXPECTED_SCOPE),
            "execution_environment": copy.deepcopy(EXPECTED_ENVIRONMENT),
            "compile_only_contract": "compile-only; no GPU/support/model/network/fallback evidence",
            "steps": steps,
            "diagnostics": [],
            "metadata": {
                "path": self.metadata_path.name,
                "sha256": sha256_file(self.metadata_path),
                "sidecar_sha256": sha256_file(self.metadata_path.with_name(self.metadata_path.name + ".sha256")),
            },
            "hashes": {**copy.deepcopy(self.metadata["hashes"]), "metadata": metadata_hash},
            "started_at": "2026-08-04T00:00:01Z",
            "finished_at": "2026-08-04T00:00:02Z",
            "duration_seconds": 1,
            "no_output_execution": True,
        }

    def write_metadata(self, refresh_sidecar: bool = True) -> None:
        self.metadata_path.write_bytes(canonical_bytes(self.metadata))
        if refresh_sidecar:
            write_sidecar(self.metadata_path)

    def write_report(self) -> None:
        self.report_path.write_bytes(canonical_bytes(self.report))
        write_sidecar(self.report_path)

    def close(self) -> None:
        self.tempdir.cleanup()


class H3PublicRuntimeContractTests(unittest.TestCase):
    def test_kernel_symbols_cover_the_full_gfx1030_direct_closure(self) -> None:
        expected_additions = {
            "sllm_attention_preprocess_headwise_norm_rope_wave32_v1",
            "sllm_deepseek_v4_moe_route_score_hash_v1",
            "sllm_deepseek_v4_moe_route_stable_group_v1",
            "sllm_elementwise_broadcast_add_bf16_fp32_v1",
            "sllm_gdn_projection_bundle_bf16_fp32_decode_v1",
            "sllm_kv_state_bf16_to_fp8_token_major_v1",
            "sllm_kv_state_bf16_to_nvfp4_token_major_v1",
            "sllm_linear_attention_column_postprocess_v2",
            "sllm_linear_attention_column_preprocess_v2",
            "sllm_linear_attention_recurrent_column_state_v2",
            "sllm_linear_attention_recurrent_gated_norm_decode_pair_v1",
            "sllm_matmul_bf16_fp32_decode_serial_rows_v1",
            "sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1",
            "sllm_matmul_bf16_fp32_prefill_short_serial_v1",
            "sllm_matmul_bf16_to_mxfp4_block32_even_v1",
            "sllm_matmul_bf16_to_nvfp4_block16_v1",
            "sllm_matmul_fp32_to_bf16_short_mixed_v1",
            "sllm_matmul_mxfp4_w4a4_block32_decode_v1",
            "sllm_matmul_mxfp4_w4a4_block32_prefill_v1",
            "sllm_matmul_nvfp4_w4a4_block16_packed_v1",
            "sllm_minimax_m3_moe_route_sigmoid_top4_v1",
            "sllm_minimax_m3_moe_route_stable_group_v1",
            "sllm_ministral3_yarn_bf16_v1",
            "sllm_mlp_gate_up_silu_bundle_bf16_fp32_decode_v1",
            "sllm_moe_down_combine_v1",
            "sllm_moe_route_bf16_stable_topk_v1",
            "sllm_moe_route_stable_group_v1",
            "sllm_moe_routed_gateup_v1",
            "sllm_moe_shared_gateup_v1",
            "sllm_rmsnorm_residual_fused_wave32_v1",
            "sllm_rmsnorm_residual_fused_wave64_v1",
            "sllm_token_selector_bf16_f32_mask_v1",
        }
        self.assertEqual(len(KERNEL_SYMBOLS), 58)
        self.assertEqual(tuple(sorted(KERNEL_SYMBOLS)), KERNEL_SYMBOLS)
        self.assertTrue(expected_additions <= set(KERNEL_SYMBOLS))
        self.assertEqual(len(expected_additions), 32)

    def test_causal_attention_stub_allowlist_is_exact_and_duplicate_free(self) -> None:
        expected = (
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_138__device_stub__causal_attention_kernelILb0EEEvPKtPKvS5_S5_S5_PKfS7_Ptjmmmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_138__device_stub__causal_attention_kernelILb1EEEvPKtPKvS5_S5_S5_PKfS7_Ptjmmmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_combine_kernelEPfPKfPKtS5_PKmS5_jjmmjjjj",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_pack_kv_kernelEPKtS2_PtS3_Pmmjjj",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_scatter_kernelEPKfPtjjjjjj",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_147__device_stub__scaled_prefill_pack_query_kernelEPKtPtPfjjjjjj",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_149__device_stub__scaled_prefill_softmax_fp16_kernelEPfPtS2_jmmjjPKff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_156__device_stub__causal_attention_decode_wave_split_kernelILb0EEEvPKtPKvS5_S5_S5_PKfS7_Ptmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_156__device_stub__causal_attention_decode_wave_split_kernelILb1EEEvPKtPKvS5_S5_S5_PKfS7_Ptmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_158__device_stub__causal_attention_prefill_gqa4_qtile4_kernelEPKtPKvS4_S4_S4_PKfS6_Ptjmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_158__device_stub__causal_attention_prefill_gqa4_shared_kernelEPKtPKvS4_S4_S4_PKfS6_Ptjmjjjjff",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_161__device_stub__causal_attention_long_prefill_v2_stage1_kernelEPKtS2_S2_jmmmjjPf",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_162__device_stub__causal_attention_long_prefill_v2_combine_kernelEPKfPtjjm",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage1_kernelILj16EEEvPKtS3_S3_PtmPf",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage1_kernelILj32EEEvPKtS3_S3_PtmPf",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage2_kernelILj16EEEvPKtPtjPKf",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage2_kernelILj32EEEvPKtPtjPKf",
            "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_166__device_stub__causal_attention_decode_wave_split_fp16_pair_kernelEPKtPKvS4_S4_S4_PKfS6_Ptmjjjjff",
        )
        self.assertEqual(len(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS), 18)
        self.assertEqual(len(set(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS)), 18)
        self.assertEqual(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS, expected)
        self.assertEqual(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS, tuple(sorted(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS)))

    def test_host_hip_undefined_closure_includes_the_three_gfx1030_additions(self) -> None:
        additions = {"hipMemRetainAllocationHandle", "hipMemcpy", "hipMemsetAsync"}
        self.assertEqual(len(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS), 52)
        self.assertEqual(len(set(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS)), 52)
        self.assertEqual(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS, tuple(sorted(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS)))
        self.assertEqual(additions, additions & set(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS))

    def test_public_symbols_are_exactly_the_umbrella_header_extern_c_set(self) -> None:
        self.assertEqual(len(PUBLIC_SYMBOLS), 109)
        self.assertEqual(declared_public_symbols(ROOT), PUBLIC_SYMBOLS)
        with tempfile.TemporaryDirectory(prefix="sllm-h3-public-symbol-header-") as directory:
            repo = Path(directory)
            header = repo / "include/sllm/hip.h"
            header.parent.mkdir(parents=True)
            source = (ROOT / "include/sllm/hip.h").read_text(encoding="utf-8")
            source = source.replace("SLLM_HIP_API sllm_status_t sllm_buffer_copy_d2d(", "SLLM_HIP_API sllm_status_t sllm_buffer_copy_d2d_removed(", 1)
            header.write_text(source, encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                declared_public_symbols(repo)

    def test_public_runtime_exact_one_job_serial_profile_and_adversarial_mutations(self) -> None:
        workflow_path = ROOT / ".github/workflows/h3-public-runtime-compile.yml"
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_h3_public_runtime_workflow(workflow_path, workflow), [])
        job = workflow["jobs"]["h3-public-runtime"]
        steps = job["steps"]
        self.assertEqual(set(workflow["jobs"]), {"h3-public-runtime"})
        self.assertEqual([step["name"] for step in steps], [
            "Checkout immutable candidate",
            "Prepare private public-H3 directories",
            "Verify immutable identity and pinned image",
            "Compile, link, extract, and inspect gfx1030",
            "Compile, link, extract, and inspect gfx1201",
            "Prepare exact public-runtime needs input",
            "Aggregate exactly two public-runtime PASS rows locally",
            "Upload JSON aggregate only",
            "Cleanup generated public-H3 rows and needs",
        ])
        self.assertIn("--row h3-public-gfx1030", steps[3]["run"])
        self.assertIn("--row h3-public-gfx1201", steps[4]["run"])
        self.assertEqual([steps[3]["name"], steps[4]["name"]], [
            "Compile, link, extract, and inspect gfx1030",
            "Compile, link, extract, and inspect gfx1201",
        ])
        self.assertLess(steps.index(next(step for step in steps if step["name"].startswith("Prepare exact"))), steps.index(next(step for step in steps if step["name"].startswith("Aggregate"))))

        def reject(mutation: object, label: str) -> None:
            with self.subTest(label=label):
                mutated = copy.deepcopy(workflow)
                mutation(mutated)
                with self.assertRaises(ManifestContractError):
                    validate_h3_public_runtime_workflow(workflow_path, mutated)

        mutations = (
            (lambda w: w["jobs"]["h3-public-runtime"].__setitem__("strategy", {"matrix": {"target": ["gfx1030", "gfx1201"]}}), "matrix topology"),
            (lambda w: w["jobs"]["h3-public-runtime"].__setitem__("needs", ["other-job"]), "needs topology"),
            (lambda w: w["jobs"].__setitem__("h3-public-gfx1030", {}), "separate row job"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"].__setitem__(4, w["jobs"]["h3-public-runtime"]["steps"][3]), "duplicate row command"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"].__setitem__(3, w["jobs"]["h3-public-runtime"]["steps"][4]), "reordered row commands"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][1].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][1]["run"].replace('test ! -e "$ROW_ROOT"\n', "")), "preexisting directory acceptance"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][1].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][1]["run"].replace('test ! -L "$ROW1030"\n', "")), "preexisting row symlink acceptance"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][1].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][1]["run"].replace("stat -c '%u:%g'", "stat -c '%a'")), "row owner check drift"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][1].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][1]["run"].replace('test "$(stat -c \'%a\' "$path")" = 700', "test \"$(stat -c '%a' \"$path\")\" = 755")), "private mode drift"),
            (lambda w: w["jobs"]["h3-public-runtime"].__setitem__("timeout-minutes", 16), "weakened job resource bound"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][2].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][2]["run"].replace('test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"\n', "")), "missing reviewed SHA"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][2].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][2]["run"].replace('test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"\n', "")), "missing immutable tree identity"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][2].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][2]["run"].replace('test -z "$(git status --porcelain=v1 --untracked-files=all)"', "true")), "missing clean checkout"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][2].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][2]["run"].replace("H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST", "WRONG_IMAGE_CONFIG_DIGEST")), "image identity drift"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("if", "${{ always() }}"), "row failure continuation"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("--network none", "--network bridge")), "network enabled"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("dst=/workspace,readonly", "dst=/workspace")), "writable source mount"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/usr/bin/git,dst=/usr/local/bin/git")), "writable helper mount"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/dev/kfd,dst=/dev/kfd")), "GPU device mount"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/dev/dri,dst=/dev/dri")), "GPU render mount"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("src=/usr/bin/git,dst=/usr/local/bin/git,readonly", "src=/var/run/docker.sock,dst=/var/run/docker.sock")), "Docker socket mount"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"].replace("--strict-ci --pinned-container", "--strict-ci")), "weakened strict CI"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"] + "\n./public-runtime-gfx1030.elf\n"), "generated executable execution"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"].insert(7, {"name": "Download row", "uses": "actions/download-artifact@" + "a" * 40}), "row download"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][3].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][3]["run"] + "\ndocker cp row /tmp/row\n"), "row transport"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][5].__setitem__("if", "${{ success() }}"), "needs not always"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][6].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][6]["run"].replace("--tree-oid", "--wrong-tree-oid")), "noncanonical aggregate"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][6].__setitem__("if", "${{ success() }}"), "aggregate not always"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][7].__setitem__("if", "${{ always() }}"), "upload on failure"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][7]["with"].__setitem__("path", ".local-artifacts"), "broad upload"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][8].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][8]["run"].replace('rm -rf -- "$ROW1030"', 'rm -rf -- "$ARTIFACT_ROOT"')), "broad cleanup"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][8].__setitem__("if", "${{ success() }}"), "cleanup not always"),
            (lambda w: w["jobs"]["h3-public-runtime"]["steps"][8].__setitem__("run", w["jobs"]["h3-public-runtime"]["steps"][8]["run"].replace('test -f "$AGGREGATE_ROOT/aggregate.json"', "")), "missing aggregate retention proof"),
        )
        for mutation, label in mutations:
            reject(mutation, label)

    def test_public_runtime_action_inputs_trigger_and_json_only_upload_are_closed(self) -> None:
        workflow_path = ROOT / ".github/workflows/h3-public-runtime-compile.yml"
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

        def assert_rejected(mutation: object, label: str) -> None:
            with self.subTest(label=label):
                mutated = copy.deepcopy(workflow)
                mutation(mutated)
                with self.assertRaises(ManifestContractError):
                    validate_h3_public_runtime_workflow(workflow_path, mutated)

        assert_rejected(lambda w: w["jobs"]["h3-public-runtime"]["steps"][0]["with"].__setitem__("ref", "main"), "mutable checkout ref")
        assert_rejected(lambda w: w["jobs"]["h3-public-runtime"]["steps"][0]["with"].__setitem__("fetch-depth", 1), "shallow checkout")
        assert_rejected(lambda w: w["jobs"]["h3-public-runtime"]["steps"][0]["with"].__setitem__("persist-credentials", True), "checkout credentials")
        assert_rejected(lambda w: w["jobs"]["h3-public-runtime"]["steps"][0].__setitem__("uses", "actions/checkout@" + "f" * 40), "arbitrary checkout action")
        assert_rejected(lambda w: w["jobs"]["h3-public-runtime"]["steps"][7].__setitem__("uses", "actions/upload-artifact@" + "f" * 40), "arbitrary upload action")
        mutated = copy.deepcopy(workflow)
        mutated["jobs"]["h3-public-runtime"]["steps"][7]["with"]["path"] += ".local-artifacts/h3-public-runtime/h3-public-gfx1030/**\n"
        with self.assertRaises(ManifestContractError):
            validate_h3_public_runtime_workflow(workflow_path, mutated)
        mutated = copy.deepcopy(workflow)
        mutated["jobs"]["h3-public-runtime"]["steps"][7]["with"]["retention-days"] = 30
        with self.assertRaises(ManifestContractError):
            validate_h3_public_runtime_workflow(workflow_path, mutated)
        mutated = copy.deepcopy(workflow)
        mutated["push"] = {"branches": ["feature"]}
        with self.assertRaises(ManifestContractError):
            validate_h3_public_runtime_workflow(workflow_path, mutated)

    def test_build_script_rerun_registration_covers_exact_h3_public_runtime_inputs(self) -> None:
        build_script = ROOT / "crates/sllm-hip-sys/build.rs"
        source = build_script.read_text(encoding="utf-8")
        _assert_h3_build_inputs_registered(build_script)
        registrations = _build_script_rerun_paths(build_script)
        binding_by_path = {path: name for name, path in registrations}
        required = set(EXPECTED_DIRECT_COMPILE_SOURCE_PATHS) | {"native/hip/CMakeLists.txt"}
        self.assertTrue(required.issubset(binding_by_path))
        for relative_path in sorted(required):
            binding = binding_by_path[relative_path]
            registration = f'println!("cargo:rerun-if-changed={{}}", {binding}.display());'
            with self.subTest(label=f"missing {relative_path}"):
                missing = source.replace(
                    f"{binding}.display()",
                    "missing_h3_build_input.display()",
                    1,
                )
                with tempfile.TemporaryDirectory(prefix="sllm-h3-build-script-missing-") as directory:
                    mutated_build_script = Path(directory) / "build.rs"
                    mutated_build_script.write_text(missing, encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        _assert_h3_build_inputs_registered(mutated_build_script)
            with self.subTest(label=f"duplicate {relative_path}"):
                duplicate = source + "\n" + registration + "\n"
                with tempfile.TemporaryDirectory(prefix="sllm-h3-build-script-duplicate-") as directory:
                    mutated_build_script = Path(directory) / "build.rs"
                    mutated_build_script.write_text(duplicate, encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        _assert_h3_build_inputs_registered(mutated_build_script)
            other_path = next(path for path in sorted(required) if path != relative_path)
            other_binding = binding_by_path[other_path]
            with self.subTest(label=f"substituted {relative_path}"):
                substituted = source.replace(
                    f"{binding}.display()",
                    f"{other_binding}.display()",
                    1,
                )
                with tempfile.TemporaryDirectory(prefix="sllm-h3-build-script-substituted-") as directory:
                    mutated_build_script = Path(directory) / "build.rs"
                    mutated_build_script.write_text(substituted, encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        _assert_h3_build_inputs_registered(mutated_build_script)

    def test_build_script_registration_parser_rejects_literal_comment_attribute_and_macro_decoys(self) -> None:
        build_script = ROOT / "crates/sllm-hip-sys/build.rs"
        source = build_script.read_text(encoding="utf-8")
        source_without_registration = source.replace(
            '    println!("cargo:rerun-if-changed={}", hip_compile_probe.display());\n',
            "",
            1,
        )
        hash_count = 37
        hashes = "#" * hash_count
        decoys = (
            (
                "ordinary string",
                r'''const _H3_ORDINARY_STRING_DECOY: &str = "println!(\"cargo:rerun-if-changed={}\", hip_compile_probe.display());";''',
            ),
            (
                "raw string without hashes",
                r'''const _H3_RAW_STRING_DECOY: &str = r"println!('decoy', hip_compile_probe.display());";''',
            ),
            (
                "raw string with one hash",
                r'''const _H3_RAW_HASH_STRING_DECOY: &str = r#"println!("cargo:rerun-if-changed={}", hip_compile_probe.display());"#;''',
            ),
            (
                "raw string with arbitrary hash count",
                f'''const _H3_RAW_MANY_HASH_STRING_DECOY: &str = r{hashes}"println!("cargo:rerun-if-changed={{}}", hip_compile_probe.display());"{hashes};''',
            ),
            (
                "ordinary byte string",
                r'''const _H3_BYTE_STRING_DECOY: &[u8] = b"println!(\"cargo:rerun-if-changed={}\", hip_compile_probe.display());";''',
            ),
            (
                "byte raw string",
                r'''const _H3_BYTE_RAW_STRING_DECOY: &[u8] = br###"println!("cargo:rerun-if-changed={}", hip_compile_probe.display());"###;''',
            ),
            (
                "character literal in macro token tree",
                r'''h3_audit_decoy! { const _H3_CHAR_DECOY: char = '"'; println!("cargo:rerun-if-changed={}", hip_compile_probe.display()); }''',
            ),
            (
                "byte character literal in macro token tree",
                r'''h3_audit_decoy! { const _H3_BYTE_CHAR_DECOY: u8 = b'\n'; println!("cargo:rerun-if-changed={}", hip_compile_probe.display()); }''',
            ),
            (
                "nested block comment",
                '''/* outer decoy comment
                   /* inner decoy: println!("cargo:rerun-if-changed={}", hip_compile_probe.display()); */
                   still inside outer comment
                */''',
            ),
            (
                "line comment",
                '// line decoy: println!("cargo:rerun-if-changed={}", hip_compile_probe.display());',
            ),
            (
                "attribute token tree",
                r'''#[h3_audit_decoy(println!("cargo:rerun-if-changed={}", hip_compile_probe.display()))]
const _H3_ATTRIBUTE_DECOY: usize = 0;''',
            ),
            (
                "macro invocation token tree",
                r'''h3_audit_decoy! {
    println!("cargo:rerun-if-changed={}", hip_compile_probe.display());
}''',
            ),
            (
                "macro_rules definition token tree",
                r'''macro_rules! h3_audit_decoy {
    () => {
        println!("cargo:rerun-if-changed={}", hip_compile_probe.display());
    };
}''',
            ),
        )
        for label, decoy in decoys:
            with self.subTest(label=label):
                mutated_source = source_without_registration + "\n" + decoy + "\n"
                with tempfile.TemporaryDirectory(prefix="sllm-h3-build-script-decoy-") as directory:
                    mutated_build_script = Path(directory) / "build.rs"
                    mutated_build_script.write_text(mutated_source, encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        _assert_h3_build_inputs_registered(mutated_build_script)

        exact_raw_string_mutation = (
            source_without_registration
            + '\nconst _H3_INDEPENDENT_RAW_STRING_DECOY: &str = '
            + 'r#"println!("cargo:rerun-if-changed={}", hip_compile_probe.display());"#;\n'
        )
        with tempfile.TemporaryDirectory(prefix="sllm-h3-build-script-raw-mutation-") as directory:
            mutated_build_script = Path(directory) / "build.rs"
            mutated_build_script.write_text(exact_raw_string_mutation, encoding="utf-8")
            with self.assertRaises(AssertionError):
                _assert_h3_build_inputs_registered(mutated_build_script)

    def test_generic_h3_owns_each_rmsnorm_public_runtime_path_and_dedicated_h3_retains_it(self) -> None:
        paths = read_json(ROOT / "ci/matrix/path-to-suite-v1.json")
        matrix_registry.validate_public_runtime_path_ownership(paths)
        matrix_registry.validate_rmsnorm_path_ownership(paths)
        generic_paths = {
            "native/hip/src/rmsnorm_api.cpp",
            "native/hip/src/rmsnorm_api.hpp",
            "native/hip/src/rmsnorm_kernel.hip.cpp",
            "native/hip/src/rmsnorm_kernel_internal.hpp",
        }
        for path in sorted(generic_paths):
            mutated = copy.deepcopy(paths)
            rule = next(rule for rule in mutated["rules"] if rule["pattern"] == path)
            rule["suite_ids"].remove(matrix_registry.H3_PUBLIC_RUNTIME_SUITE_ID)
            with self.subTest(path=path):
                with self.assertRaises(matrix_registry.ContractError):
                    matrix_registry.validate_public_runtime_path_ownership(mutated)
                matrix_registry.validate_rmsnorm_path_ownership(mutated)

    def test_rust_registration_tokenizer_fails_closed_on_unterminated_literals_and_comments(self) -> None:
        malformed = (
            ("ordinary string", '"'),
            ("ordinary byte string", 'b"'),
            ("raw string", 'r#"unterminated'),
            ("byte raw string", 'br##"unterminated'),
            ("character literal", "'"),
            ("character escape", "'" + "\\"),
            ("byte character literal", "b'"),
            ("nested block comment", "/* outer /* inner */"),
        )
        for label, malformed_source in malformed:
            with self.subTest(label=label):
                with self.assertRaises(AssertionError):
                    _rust_tokens(malformed_source)

    def test_static_contract_and_artifact_fixture_pass_without_rocm_or_execution(self) -> None:
        validate_static(ROOT)
        for target in TARGETS:
            with self.subTest(target=target):
                fixture = ArtifactFixture(target)
                try:
                    metadata = validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
                    report, _report_sha, _sidecar_sha = validate_report(fixture.report_path, metadata, ROOT, fixture.row_dir)
                    self.assertEqual(report["state"], "PASS")
                    self.assertEqual(metadata["target"], target)
                    self.assertTrue(all("{" not in token and "}" not in token for command in metadata["build"]["commands"] for token in command))
                finally:
                    fixture.close()

    def test_rendered_metadata_commands_reject_template_target_filename_and_order_mutations(self) -> None:
        def replace_token(commands: list[list[str]], source: str, replacement: str) -> None:
            for command in commands:
                if source in command:
                    command[command.index(source)] = replacement
                    return
            raise AssertionError(f"missing fixture command token: {source}")

        def reorder_argv(metadata: dict[str, object]) -> None:
            command = metadata["build"]["commands"][0]
            command[1], command[2] = command[2], command[1]

        def reorder_commands(metadata: dict[str, object]) -> None:
            commands = metadata["build"]["commands"]
            commands[0], commands[1] = commands[1], commands[0]

        mutations = (
            (
                lambda metadata: replace_token(
                    metadata["build"]["commands"],
                    "--offload-arch=gfx1030",
                    "--offload-arch=gfx1201",
                ),
                "wrong target",
            ),
            (
                lambda metadata: replace_token(
                    metadata["build"]["commands"],
                    "--offload-arch=gfx1030",
                    "--offload-arch={target}",
                ),
                "literal target template",
            ),
            (
                lambda metadata: replace_token(
                    metadata["build"]["commands"],
                    "/proc/self/fd/5/hip-compile-probe-gfx1030.o",
                    "/proc/self/fd/5/hip-compile-probe-gfx1201.o",
                ),
                "wrong target filename",
            ),
            (
                lambda metadata: replace_token(
                    metadata["build"]["commands"],
                    str(ROOT) + "/include",
                    str(ROOT) + "/include.suffix",
                ),
                "repository suffix",
            ),
            (
                lambda metadata: replace_token(
                    metadata["build"]["commands"],
                    "/proc/self/fd/5/public-runtime-gfx1030.o",
                    "/proc/self/fd/5/public-runtime-gfx1030.o.suffix",
                ),
                "build filename suffix",
            ),
            (
                lambda metadata: metadata["build"]["commands"][0].append("--unexpected"),
                "extra argv token",
            ),
            (reorder_argv, "reordered argv"),
            (reorder_commands, "reordered command"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture("gfx1030")
                try:
                    mutation(fixture.metadata)
                    fixture.write_metadata()
                    with self.assertRaises(ContractError):
                        validate_metadata(
                            fixture.metadata_path,
                            ROOT,
                            expected_sha=fixture.identity["commit_sha"],
                            expected_tree=fixture.identity["tree_oid"],
                            artifact_root=fixture.row_dir,
                        )
                finally:
                    fixture.close()

    def test_report_steps_are_exactly_bound_to_rendered_metadata_commands(self) -> None:
        def reorder_steps(report: dict[str, object]) -> None:
            steps = report["steps"]
            steps[0], steps[1] = steps[1], steps[0]

        mutations = (
            (lambda report: report["steps"][0]["argv"].append("--unexpected"), "argv mutation"),
            (lambda report: report["steps"][0].__setitem__("step_id", "h3-public-gfx1030.compile-2"), "step ID mutation"),
            (reorder_steps, "step order mutation"),
            (lambda report: report["steps"][0].__setitem__("exit_code", 1), "exit code mutation"),
            (lambda report: report["steps"][0]["resource"].__setitem__("timed_out", True), "timeout mutation"),
            (lambda report: report["steps"][0].__setitem__("diagnostic", "unexpected"), "diagnostic mutation"),
            (lambda report: report["steps"][0]["resource"].__setitem__("output_limit_bytes", 1), "output limit drift"),
            (lambda report: report["steps"][0]["resource"].__setitem__("max_rss_limit_bytes", 1), "RSS limit drift"),
            (lambda report: report["steps"][0]["resource"].__setitem__("output_bytes", 16777217), "output limit exceedance"),
            (lambda report: report["steps"][0]["resource"].__setitem__("max_rss_bytes", 4294967297), "RSS limit exceedance"),
            (lambda report: report["steps"][0].__setitem__("started_at", "2026-08-04T00:00:01.200Z"), "reversed step timestamps"),
            (lambda report: report["steps"][1].__setitem__("started_at", "2026-08-04T00:00:01.050Z"), "step chronology mutation"),
            (lambda report: report["steps"][0].__setitem__("started_at", "2026-08-04T00:00:00.999Z"), "report timestamp bound mutation"),
            (lambda report: report["steps"][0].__setitem__("duration_seconds", float("inf")), "nonfinite duration"),
            (lambda report: report["steps"][0].__setitem__("stdout_sha256", "A" * 64), "noncanonical digest"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture("gfx1030")
                try:
                    mutation(fixture.report)
                    fixture.write_report()
                    with self.assertRaises(ContractError):
                        validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
                finally:
                    fixture.close()

    def test_static_only_and_collection_root_one_row_validation_pass(self) -> None:
        validate_contracts(ROOT, [])
        self.assertEqual(validate_main(["--repo", str(ROOT)]), 0)

        fixture = ArtifactFixture()
        collection = Path(tempfile.mkdtemp(prefix="sllm-h3-public-runtime-one-row-"))
        try:
            row_root = collection / fixture.row_id
            shutil.copytree(fixture.row_dir, row_root)
            validate_contracts(
                ROOT,
                [row_root / "hip-runtime-artifact.json"],
                expected_sha=fixture.identity["commit_sha"],
                expected_tree=fixture.identity["tree_oid"],
                artifact_root=collection,
            )
        finally:
            fixture.close()
            shutil.rmtree(collection)

    def test_collection_root_main_discovers_exactly_two_direct_rows(self) -> None:
        collection = Path(tempfile.mkdtemp(prefix="sllm-h3-public-runtime-two-row-"))
        fixtures = [ArtifactFixture("gfx1030"), ArtifactFixture("gfx1201")]
        try:
            for fixture in fixtures:
                shutil.copytree(fixture.row_dir, collection / fixture.row_id)
            self.assertEqual(
                validate_main(
                    [
                        "--repo",
                        str(ROOT),
                        "--artifact-root",
                        str(collection),
                        "--expected-candidate-sha",
                        fixtures[0].identity["commit_sha"],
                        "--expected-tree-oid",
                        fixtures[0].identity["tree_oid"],
                    ]
                ),
                0,
            )
        finally:
            for fixture in fixtures:
                fixture.close()
            shutil.rmtree(collection)

    def test_collection_root_boundaries_fail_closed(self) -> None:
        fixture = ArtifactFixture()
        collection = Path(tempfile.mkdtemp(prefix="sllm-h3-public-runtime-boundary-"))
        try:
            row_root = collection / fixture.row_id
            shutil.copytree(fixture.row_dir, row_root)
            identity = {
                "expected_sha": fixture.identity["commit_sha"],
                "expected_tree": fixture.identity["tree_oid"],
                "artifact_root": collection,
            }

            unexpected_file = collection / "unexpected.top"
            unexpected_file.write_text("unexpected\n", encoding="utf-8")
            with self.subTest("unexpected regular file at collection root"):
                self.assertEqual(
                    validate_main(
                        [
                            "--repo",
                            str(ROOT),
                            "--artifact-root",
                            str(collection),
                            "--expected-candidate-sha",
                            fixture.identity["commit_sha"],
                            "--expected-tree-oid",
                            fixture.identity["tree_oid"],
                        ]
                    ),
                    1,
                )
            unexpected_file.unlink()

            unexpected_directory = collection / "unexpected-directory"
            unexpected_directory.mkdir()
            with self.subTest("unexpected directory at collection root"):
                self.assertEqual(
                    validate_main(
                        [
                            "--repo",
                            str(ROOT),
                            "--artifact-root",
                            str(collection),
                            "--expected-candidate-sha",
                            fixture.identity["commit_sha"],
                            "--expected-tree-oid",
                            fixture.identity["tree_oid"],
                        ]
                    ),
                    1,
                )
            unexpected_directory.rmdir()

            with self.subTest("outside root"):
                with self.assertRaises(ContractError):
                    validate_contracts(ROOT, [fixture.metadata_path], **identity)

            nested = collection / "nested" / fixture.row_id
            shutil.copytree(fixture.row_dir, nested)
            with self.subTest("nested same-basename metadata"):
                with self.assertRaises(ContractError):
                    validate_contracts(ROOT, [nested / "hip-runtime-artifact.json"], **identity)

            with self.subTest("duplicate metadata path"):
                with self.assertRaises(ContractError):
                    validate_contracts(
                        ROOT,
                        [row_root / "hip-runtime-artifact.json"] * 2,
                        **identity,
                    )

            symlink_row = collection / "h3-public-gfx1201"
            symlink_row.symlink_to(fixture.row_dir, target_is_directory=True)
            with self.subTest("symlink row"):
                self.assertEqual(
                    validate_main(
                        [
                            "--repo",
                            str(ROOT),
                            "--artifact-root",
                            str(collection),
                            "--expected-candidate-sha",
                            fixture.identity["commit_sha"],
                            "--expected-tree-oid",
                            fixture.identity["tree_oid"],
                        ]
                    ),
                    1,
                )
            symlink_row.unlink()

            symlink_root = collection.parent / (collection.name + "-symlink")
            symlink_root.symlink_to(collection, target_is_directory=True)
            try:
                with self.subTest("symlink collection root"):
                    self.assertEqual(
                        validate_main(
                            [
                                "--repo",
                                str(ROOT),
                                "--artifact-root",
                                str(symlink_root),
                                "--expected-candidate-sha",
                                fixture.identity["commit_sha"],
                                "--expected-tree-oid",
                                fixture.identity["tree_oid"],
                            ]
                        ),
                        1,
                    )
            finally:
                symlink_root.unlink()
        finally:
            fixture.close()
            shutil.rmtree(collection)

        empty = Path(tempfile.mkdtemp(prefix="sllm-h3-public-runtime-empty-"))
        try:
            with self.subTest("empty explicit root"):
                self.assertEqual(
                    validate_main(["--repo", str(ROOT), "--artifact-root", str(empty)]),
                    1,
                )
        finally:
            shutil.rmtree(empty)

    def test_validate_report_rejects_every_unexpected_row_root_entry(self) -> None:
        mutations = (
            (lambda fixture: (fixture.row_dir / "unexpected-file").write_text("unexpected\n", encoding="utf-8"), "unexpected row-root file"),
            (lambda fixture: (fixture.row_dir / "unexpected-directory").mkdir(), "unexpected row-root directory"),
            (lambda fixture: (fixture.row_dir / "unexpected-symlink").symlink_to(fixture.output_paths["host_elf"]), "unexpected row-root symlink"),
            (lambda fixture: (fixture.build_dir / "unexpected-directory").mkdir(), "unexpected build directory"),
            (lambda fixture: (fixture.build_dir / "unexpected-symlink").symlink_to(fixture.output_paths["host_elf"]), "unexpected build symlink"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture()
                try:
                    mutation(fixture)
                    with self.assertRaises(ContractError):
                        validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
                finally:
                    fixture.close()

    def test_report_hashes_require_exact_metadata_and_output_associations(self) -> None:
        mutations: list[tuple[object, str]] = [
            (lambda report: report["hashes"].pop("metadata"), "missing metadata hash"),
            (lambda report: report["hashes"].pop("probe_object"), "missing output hash"),
            (lambda report: report["hashes"].__setitem__("unexpected", copy.deepcopy(report["hashes"]["metadata"])), "unexpected hash key"),
        ]
        for field in ("path", "size_bytes", "sha256", "sidecar_path", "sidecar_sha256"):
            mutations.append(
                (
                    lambda report, field=field: report["hashes"]["probe_object"].__setitem__(
                        field,
                        "/output/build/wrong.o" if field in {"path", "sidecar_path"} else (0 if field == "size_bytes" else "0" * 64),
                    ),
                    f"stale probe-object {field}",
                )
            )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture()
                try:
                    mutation(fixture.report)
                    fixture.write_report()
                    with self.assertRaises(ContractError):
                        validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
                finally:
                    fixture.close()

        fixture = ArtifactFixture()
        try:
            self.assertEqual(set(fixture.report["hashes"]), {"metadata", *fixture.metadata["hashes"]})
            metadata = validate_metadata(
                fixture.metadata_path,
                ROOT,
                expected_sha=fixture.identity["commit_sha"],
                expected_tree=fixture.identity["tree_oid"],
                artifact_root=fixture.row_dir,
            )
            validate_report(fixture.report_path, metadata, ROOT, fixture.row_dir)
        finally:
            fixture.close()

    def test_artifact_identity_and_staged_paths_are_strictly_bound(self) -> None:
        fixture = ArtifactFixture()
        try:
            with self.assertRaises(ContractError):
                validate_metadata(fixture.metadata_path, ROOT, artifact_root=fixture.row_dir)
            with self.assertRaises(ContractError):
                validate_metadata(fixture.metadata_path, ROOT, expected_sha="c" * 40, expected_tree="d" * 40, artifact_root=fixture.row_dir)
            fixture.metadata["candidate"]["commit_sha"] = "c" * 40
            fixture.write_metadata()
            with self.assertRaises(ContractError):
                validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
        finally:
            fixture.close()

        fixture = ArtifactFixture()
        try:
            name = "probe_object"
            record = fixture.metadata["hashes"][name]
            staged_elsewhere = fixture.row_dir / "elsewhere" / Path(record["path"]).name
            staged_elsewhere.parent.mkdir()
            staged_elsewhere.write_bytes(fixture.output_paths[name].read_bytes())
            staged_elsewhere.with_name(staged_elsewhere.name + ".sha256").write_text(
                f"{sha256_file(staged_elsewhere)}  {staged_elsewhere.name}\n", encoding="ascii"
            )
            record["path"] = "/output/elsewhere/" + staged_elsewhere.name
            record["sidecar_path"] = record["path"] + ".sha256"
            fixture.write_metadata()
            with self.assertRaises(ContractError):
                validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
        finally:
            fixture.close()

    def test_required_report_metadata_sidecars_and_association_fail_closed(self) -> None:
        fixture = ArtifactFixture()
        try:
            metadata_args = {
                "expected_sha": fixture.identity["commit_sha"],
                "expected_tree": fixture.identity["tree_oid"],
                "artifact_root": fixture.row_dir,
            }
            fixture.metadata_path.with_name("hip-runtime-artifact.json.sha256").unlink()
            with self.assertRaises(ContractError):
                validate_metadata(fixture.metadata_path, ROOT, **metadata_args)
        finally:
            fixture.close()

        fixture = ArtifactFixture()
        try:
            fixture.report_path.with_name("report.json.sha256").unlink()
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
        finally:
            fixture.close()

        fixture = ArtifactFixture()
        try:
            fixture.report["metadata"]["path"] = "wrong-metadata.json"
            fixture.write_report()
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
        finally:
            fixture.close()

        fixture = ArtifactFixture()
        try:
            fixture.report_path.unlink()
            with self.assertRaises(ContractError):
                validate_contracts(
                    ROOT,
                    [fixture.metadata_path],
                    expected_sha=fixture.identity["commit_sha"],
                    expected_tree=fixture.identity["tree_oid"],
                    artifact_root=fixture.row_dir,
                )
        finally:
            fixture.close()

    def test_report_schema_is_registered_closed_and_probe_symbol_is_exact(self) -> None:
        validate_static(ROOT)
        with patch.dict(sys.modules, {"jsonschema": None}):
            with self.assertRaises(ContractError):
                validate_static(ROOT)
        fixture = ArtifactFixture()
        try:
            fixture.report["unexpected"] = True
            fixture.write_report()
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
            duplicate_fixture = ArtifactFixture()
            try:
                duplicate_fixture.metadata["device_code_object"]["symbols"].append({"name": "sllm_hip_compile_probe", "defined": True})
                duplicate_fixture.write_metadata()
                with self.assertRaises((ContractError, RuntimeContractError)):
                    validate_metadata(duplicate_fixture.metadata_path, ROOT, expected_sha=duplicate_fixture.identity["commit_sha"], expected_tree=duplicate_fixture.identity["tree_oid"], artifact_root=duplicate_fixture.row_dir)
            finally:
                duplicate_fixture.close()
        finally:
            fixture.close()

        temp = Path(tempfile.mkdtemp(prefix="sllm-h3-report-schema-missing-"))
        try:
            shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
            (temp / "ci/schema/hip-runtime-public-report-v1.schema.json").unlink()
            with self.assertRaises((ContractError, RuntimeContractError)):
                validate_static(temp)
        finally:
            shutil.rmtree(temp)

    def test_pass_schema_keeps_exact_environment_and_all_seven_hash_associations(self) -> None:
        schema = read_json(ROOT / "ci/schema/hip-runtime-public-report-v1.schema.json")
        mutations = (
            (lambda report: report["execution_environment"].__setitem__("identity_verified", False), "relaxed PASS environment"),
            (lambda report: report.__setitem__("metadata", None), "missing PASS metadata association"),
            (lambda report: report["hashes"].pop("device_object"), "missing PASS device association"),
            (lambda report: report["hashes"].pop("rmsnorm_kernel_object"), "missing PASS kernel association"),
            (lambda report: report["hashes"].pop("rmsnorm_api_object"), "missing PASS API association"),
            (lambda report: report["hashes"].__setitem__("unexpected", copy.deepcopy(report["hashes"]["metadata"])), "extra PASS hash association"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture()
                try:
                    mutation(fixture.report)
                    with self.assertRaises(ContractError):
                        validate_against_schema(fixture.report, schema, "PASS report")
                finally:
                    fixture.close()

    def test_dirty_checkout_failure_report_is_schema_valid_before_static_contracts(self) -> None:
        temp = Path(tempfile.mkdtemp(prefix="sllm-h3-dirty-failure-report-"))
        repo = temp / "repo"
        output = temp / "output"
        repo.mkdir()
        try:
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "audit@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "audit"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "base", "-q"], check=True)
            (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            self.assertEqual(
                run_h3_public_runtime_main(
                    [
                        "--row",
                        "h3-public-gfx1030",
                        "--repo",
                        str(repo),
                        "--output-dir",
                        str(output),
                    ]
                ),
                1,
            )
            report = read_json(output / "report.json")
            validate_against_schema(
                report,
                read_json(ROOT / "ci/schema/hip-runtime-public-report-v1.schema.json"),
                "dirty-checkout FAIL report",
            )
            self.assertEqual(report["state"], "FAIL")
            self.assertIsNone(report["metadata"])
            self.assertEqual(report["hashes"], {})
        finally:
            shutil.rmtree(temp)

    def test_missing_changed_extra_symlinked_and_stale_canonical_source_are_rejected(self) -> None:
        for mode in ("missing", "changed", "symlinked"):
            with self.subTest(mode=mode):
                temp = Path(tempfile.mkdtemp(prefix="sllm-h3-source-") )
                try:
                    shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
                    source = temp / "native/hip/src/public_runtime.hip.cpp"
                    if mode == "missing":
                        source.unlink()
                    elif mode == "changed":
                        source.write_bytes(source.read_bytes() + b"changed\n")
                    else:
                        source.unlink()
                        source.symlink_to(temp / "native/hip/src/public_runtime_stub.cpp")
                    with self.assertRaises((ContractError, RuntimeContractError)):
                        validate_static(temp)
                finally:
                    shutil.rmtree(temp)

        temp = Path(tempfile.mkdtemp(prefix="sllm-h3-source-extra-"))
        try:
            shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
            matrix_path = temp / "ci/matrix/hip-runtime-compile-v1.json"
            matrix = json.loads(matrix_path.read_text())
            matrix["sources"]["canonical_order"].append("native/hip/src/extra.hpp")
            matrix["sources"]["files"].append({"path": "native/hip/src/extra.hpp", "sha256": "0" * 64})
            matrix_path.write_bytes(canonical_bytes(matrix))
            with self.assertRaises((ContractError, RuntimeContractError)):
                validate_static(temp)
        finally:
            shutil.rmtree(temp)

        temp = Path(tempfile.mkdtemp(prefix="sllm-h3-source-set-") )
        try:
            shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
            matrix_path = temp / "ci/matrix/hip-runtime-compile-v1.json"
            matrix = json.loads(matrix_path.read_text())
            matrix["sources"]["source_set_sha256"] = "0" * 64
            matrix_path.write_bytes(canonical_bytes(matrix))
            with self.assertRaises((ContractError, RuntimeContractError)):
                validate_static(temp)
        finally:
            shutil.rmtree(temp)

    def test_direct_compile_inventory_and_rmsnorm_symbols_reject_missing_extra_order_duplicate_and_stale(self) -> None:
        def assert_invalid(mutator: object, label: str) -> None:
            temp = Path(tempfile.mkdtemp(prefix="sllm-h3-direct-inventory-"))
            try:
                shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
                matrix_path = temp / "ci/matrix/hip-runtime-compile-v1.json"
                matrix = json.loads(matrix_path.read_text())
                mutator(matrix)
                matrix_path.write_bytes(canonical_bytes(matrix))
                with self.assertRaises((ContractError, RuntimeContractError), msg=label):
                    validate_static(temp)
            finally:
                shutil.rmtree(temp)

        assert_invalid(
            lambda matrix: matrix.__setitem__("revision", 1),
            "matrix revision drift",
        )
        assert_invalid(
            lambda matrix: matrix["rows"][0]["resource"].__setitem__("timeout_seconds", 901),
            "weakened command timeout bound",
        )
        assert_invalid(
            lambda matrix: matrix["rows"][1]["resource"].__setitem__("max_rss_bytes", 8589934592),
            "weakened command RSS bound",
        )
        assert_invalid(
            lambda matrix: matrix["sources"]["files"][0].__setitem__("unexpected", True),
            "unknown legacy source item field",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"]["files"][0].__setitem__("unexpected", True),
            "unknown direct source item field",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"]["files"].pop(),
            "missing direct source",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"]["canonical_order"].append("native/hip/src/extra.hpp"),
            "extra direct source",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"]["canonical_order"].__setitem__(0, matrix["direct_compile_sources"]["canonical_order"][1]),
            "duplicate direct source order",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"]["canonical_order"].reverse(),
            "reordered direct source inventory",
        )
        assert_invalid(
            lambda matrix: next(item for item in matrix["direct_compile_sources"]["files"] if item["path"] == "native/hip/src/rmsnorm_kernel.hip.cpp").__setitem__("sha256", "0" * 64),
            "stale RMSNorm kernel source hash",
        )
        assert_invalid(
            lambda matrix: next(item for item in matrix["direct_compile_sources"]["files"] if item["path"] == "native/hip/src/rmsnorm_kernel_internal.hpp").__setitem__("sha256", "0" * 64),
            "stale RMSNorm kernel header hash",
        )
        assert_invalid(
            lambda matrix: next(item for item in matrix["direct_compile_sources"]["files"] if item["path"] == "native/hip/src/rmsnorm_api.cpp").__setitem__("sha256", "0" * 64),
            "stale RMSNorm API source hash",
        )
        assert_invalid(
            lambda matrix: next(item for item in matrix["direct_compile_sources"]["files"] if item["path"] == "native/hip/src/rmsnorm_api.hpp").__setitem__("sha256", "0" * 64),
            "stale RMSNorm API header hash",
        )
        assert_invalid(
            lambda matrix: matrix["direct_compile_sources"].__setitem__("source_set_sha256", "0" * 64),
            "stale direct source-set hash",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].remove("sllm_rmsnorm_prepare"),
            "missing RMSNorm public symbol",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].append("sllm_rmsnorm_execute"),
            "duplicate RMSNorm execute public symbol",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].remove("sllm_rmsnorm_execute"),
            "missing RMSNorm execute public symbol",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].__setitem__(matrix["public_abi_symbols"].index("sllm_rmsnorm_execute"), "sllm_rmsnorm_substituted"),
            "substituted RMSNorm execute public symbol",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].reverse(),
            "reordered public symbol set",
        )
        assert_invalid(
            lambda matrix: matrix["public_abi_symbols"].__setitem__(0, matrix["public_abi_symbols"][1]),
            "duplicate public symbol",
        )

        def assert_invalid_schema(mutator: object, label: str) -> None:
            temp = Path(tempfile.mkdtemp(prefix="sllm-h3-compile-schema-"))
            try:
                shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
                schema_path = temp / "ci/schema/hip-runtime-compile-v1.schema.json"
                schema = json.loads(schema_path.read_text())
                mutator(schema)
                schema_path.write_bytes(canonical_bytes(schema))
                with self.assertRaises((ContractError, RuntimeContractError), msg=label):
                    validate_static(temp)
            finally:
                shutil.rmtree(temp)

        assert_invalid_schema(
            lambda schema: schema["properties"]["public_abi_symbols"].pop("const"),
            "loosened public symbol schema",
        )
        assert_invalid_schema(
            lambda schema: schema["properties"]["public_abi_symbols"].__setitem__("const", list(PUBLIC_SYMBOLS[:-1])),
            "mismatched public symbol schema",
        )

    def test_metadata_output_host_device_and_public_symbol_evidence_fail_closed(self) -> None:
        for output_name in ("probe_object", "public_runtime_object", "rmsnorm_kernel_object", "rmsnorm_api_object", "host_elf", "device_object"):
            fixture = ArtifactFixture()
            try:
                fixture.output_paths[output_name].write_bytes(b"stale-output")
                with self.assertRaises(ContractError):
                    validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
            finally:
                fixture.close()

        for output_name in ("rmsnorm_kernel_object", "rmsnorm_api_object"):
            fixture = ArtifactFixture()
            try:
                fixture.output_paths[output_name].with_name(fixture.output_paths[output_name].name + ".sha256").unlink()
                with self.assertRaises(ContractError):
                    validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
            finally:
                fixture.close()

        mutations = (
            (lambda m: m["host_elf"]["stub_symbols"].append("sllm_public_runtime_stub"), "stub-linked artifact"),
            (lambda m: m["host_elf"]["public_symbols"].pop(), "missing public C ABI symbol"),
            (lambda m: m["host_elf"].pop("kernel_symbol"), "missing linked RMSNorm kernel symbol"),
            (lambda m: m["host_elf"].update({"kernel_symbol": {"name": "sllm_rmsnorm_baseline_wave32_v1", "defined": False}}), "undefined linked RMSNorm kernel symbol"),
            (lambda m: m["host_elf"].update({"kernel_symbol": {"name": "sllm_wrong_kernel", "defined": True}}), "substituted linked RMSNorm kernel symbol"),
            (lambda m: m["device_code_object"]["symbols"].append({"name": "sllm_context_create", "defined": True}), "public symbol in device object"),
            (lambda m: m["device_code_object"]["symbols"].append({"name": "sllm_hip_compile_probe", "defined": True}), "duplicate probe symbol"),
            (lambda m: m["device_code_object"]["symbols"].clear(), "missing probe symbol"),
            (lambda m: m["device_code_object"]["symbols"].__setitem__(0, {"name": "sllm_hip_compile_probe", "defined": False}), "undefined probe symbol"),
            (lambda m: m["device_code_object"]["symbols"].__setitem__(0, {"name": "sllm_unknown", "defined": True}), "unknown device symbol"),
            (lambda m: m["device_code_object"]["symbols"].__setitem__(0, {"name": "sllm_hip_compile_probe"}), "malformed symbol record"),
            (lambda m: m["device_code_object"]["symbols"].__setitem__(0, {"name": "sllm_hip_compile_probe", "defined": True, "extra": "rejected"}), "extra symbol field"),
            (lambda m: m["device_code_object"].update({"source_attribution": "public_runtime.hip.cpp"}), "wrong device source"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                fixture = ArtifactFixture()
                try:
                    mutation(fixture.metadata)
                    fixture.write_metadata()
                    with self.assertRaises(ContractError):
                        validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
                finally:
                    fixture.close()

    def test_generic_build_rejects_wrong_tu_object_order_compiler_stub_and_extra_link_inputs(self) -> None:
        def swap_kernel_api_objects(row: dict[str, object]) -> None:
            link = row["build"]["commands"][4]
            link[12], link[13] = link[13], link[12]

        mutations = (
            (lambda row: row["build"]["commands"][2].__setitem__(-1, "{repo}/native/hip/src/public_runtime_stub.cpp"), "stub TU"),
            (lambda row: row["build"]["commands"][2].__setitem__(-1, "{repo}/native/hip/src/rmsnorm_api.cpp"), "wrong kernel TU"),
            (lambda row: row["build"]["commands"][3].__setitem__(0, "/usr/bin/c++"), "wrong API compiler"),
            (lambda row: row["build"]["commands"][4].__setitem__(12, row["build"]["commands"][4][13]), "omitted kernel object"),
            (lambda row: row["build"]["commands"][4].__setitem__(13, row["build"]["commands"][4][12]), "omitted API object"),
            (swap_kernel_api_objects, "reordered kernel/API objects"),
            (lambda row: row["build"]["commands"][4].append("{build_dir}/unexpected.o"), "unexpected link input"),
            (lambda row: row["build"]["commands"][4].__setitem__(12, "{repo}/native/hip/src/rmsnorm_kernel.hip.cpp"), "raw-source link"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                temp = Path(tempfile.mkdtemp(prefix="sllm-h3-build-shape-"))
                try:
                    shutil.copytree(ROOT, temp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".local-artifacts", "__pycache__", "target"))
                    matrix_path = temp / "ci/matrix/hip-runtime-compile-v1.json"
                    matrix = json.loads(matrix_path.read_text())
                    mutation(next(row for row in matrix["rows"] if row["target"] == "gfx1030"))
                    matrix_path.write_bytes(canonical_bytes(matrix))
                    with self.assertRaises((ContractError, RuntimeContractError)):
                        validate_static(temp)
                finally:
                    shutil.rmtree(temp)

    def test_codegen_identity_scope_and_hash_sidecars_fail_closed(self) -> None:
        for mutation, label in (
            (lambda m: m["codegen"].update({"target": "gfx1201"}), "wrong target"),
            (lambda m: m["codegen"].update({"target_count": 2}), "multiple target"),
            (lambda m: m["codegen"].update({"target_kind": "generic"}), "generic target"),
            (lambda m: m["codegen"].update({"code_object_version": "V5"}), "wrong V6"),
            (lambda m: m["codegen"].update({"wavefront_size": 64}), "wrong wave32"),
            (lambda m: m["codegen"]["features"].update({"xnack": "+"}), "wrong features"),
            (lambda m: m["codegen"].update({"e_flags": "0x0000004e"}), "wrong e_flags"),
            (lambda m: m["build"]["commands"][0].append("--offload-arch=gfx1030"), "multiple command targets"),
            (lambda m: m["build"]["commands"][0].__setitem__(1, "--offload-arch=gfx12-generic"), "generic command target"),
            (lambda m: m["build"].update({"output_directory": str(ROOT)}), "source-tree output"),
            (lambda m: m["candidate"].update({"tested_sha": "c" * 40}), "identity mismatch"),
        ):
            with self.subTest(label=label):
                fixture = ArtifactFixture()
                try:
                    mutation(fixture.metadata)
                    fixture.write_metadata()
                    with self.assertRaises(ContractError):
                        validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
                finally:
                    fixture.close()

        for key in EXPECTED_SCOPE:
            fixture = ArtifactFixture()
            try:
                fixture.metadata["scope"][key] = not EXPECTED_SCOPE[key]
                fixture.write_metadata()
                with self.assertRaises(ContractError):
                    validate_metadata(fixture.metadata_path, ROOT, expected_sha=fixture.identity["commit_sha"], expected_tree=fixture.identity["tree_oid"], artifact_root=fixture.row_dir)
            finally:
                fixture.close()

        fixture = ArtifactFixture()
        try:
            fixture.report["metadata"]["sha256"] = "0" * 64
            fixture.write_report()
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
            fixture.report = fixture._make_report()
            fixture.write_report()
            fixture.report_path.with_name(fixture.report_path.name + ".sha256").write_text("0" * 64 + "  report.json\n", encoding="ascii")
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
            fixture.report = fixture._make_report()
            fixture.write_report()
            fixture.metadata_path.with_name(fixture.metadata_path.name + ".sha256").write_text("0" * 64 + "  hip-runtime-artifact.json\n", encoding="ascii")
            with self.assertRaises(ContractError):
                validate_report(fixture.report_path, fixture.metadata, ROOT, fixture.row_dir)
        finally:
            fixture.close()

    def test_strict_dirty_identity_and_g1_cannot_use_h3_artifact(self) -> None:
        temp = Path(tempfile.mkdtemp(prefix="sllm-h3-dirty-identity-"))
        repo = temp / "repo"
        repo.mkdir()
        try:
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "audit@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "audit"], check=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "base", "-q"], check=True)
            require_clean_checkout(repo)

            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                require_clean_checkout(repo)
        finally:
            shutil.rmtree(temp)

        from validate_g1_contracts import _tail_is_dedicated_binary

        with self.assertRaises(ValueError):
            _tail_is_dedicated_binary("/tmp/h3-public-gfx1030/device-code-object-gfx1030.elf", "G1 binary")


if __name__ == "__main__":
    unittest.main()
