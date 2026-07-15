#!/usr/bin/env python3
"""Prepare and independently validate the AQ4 P2 resident one-case smoke bundle."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v2"
BINDING_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v7"
BINDING_VALIDATOR_EXEC = ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py"
SERVED_PATH = Path("/etc/ullm/served-models/active.json")
CASE_MANIFEST_PATH = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
DRIVER_BUILD_PATH = ROOT / "target/release/ullm-aq4-p2-resident-driver"
WORKER_HARDLINK_FIXTURE_PATH = ROOT / "tests/fixtures/aq4-p2-resident-worker-hardlinks/active-production.json"
SOURCE_COMMIT = "43ba16f2347a45caba8a60cac2189714118db280"
SOURCE_TREE = "72392a7114f5968d6c2ad05e24762a6790000013"
DRIVER_COMMIT = "43ba16f2347a45caba8a60cac2189714118db280"
DRIVER_TREE = "72392a7114f5968d6c2ad05e24762a6790000013"
DRIVER_SOURCE_GIT_BLOB = "7e37119cc8b66dc0e0f7abcf49b896fcdad8315f"
DRIVER_SOURCE_SHA = "0acb46d1ab8730267edf40b505224ff157760ec19aa40a07ee1b389860ec54bf"
DRIVER_BUILD_INPUTS = {
    ".cargo/config.toml": {
        "git_blob": "6dee7973a174f5e45c5762d91522f5d6849a5b84",
        "sha256": "41627bf0cfcb00817cd6bee0285a01d25c89614bb271798139c26539c525f67d",
        "role": "cargo_linker_configuration",
    },
    "Cargo.lock": {
        "git_blob": "fb12cb0388ea1c6fc6368e7ea5d5100c11a20666",
        "sha256": "10df8371ae3a33ed792dc4e8c15dd6196a8a7e176e377ef275e75b3219aa157b",
        "role": "locked_dependency_graph",
    },
    "crates/ullm-runtime-sys/build.rs": {
        "git_blob": "bfd7a966b465e6f61189ae7cae8432065f102b6f",
        "sha256": "e2d29a16e4e6be98e8cc5f41f7350e8210d707229a204fb7c0b35a9ef0d096ea",
        "role": "runtime_static_library_build",
    },
    "runtime/src/ullm_runtime_parts/part_00.inc": {
        "git_blob": "316d3ae5c13f79678fb8256aa8c66ea7e154660f",
        "sha256": "db138bfaf33f59708f24edbec8352a39fe809ff39422d5b742399752c8fa9f5f",
        "role": "directional_hip_copy_runtime",
    },
}
RUNNER_COMMIT = "3dc4aa612b6cfd87675d0bd9fe506426f43e64f9"
RUNNER_TREE = "bd46e713c658878e66fcab6d49ef863e43a06bd8"
RUNNER_SOURCE_SHA = "e7dae31c64b3844a09fbba7ef36bbae7834e21d5d217bad679dd50bdf314ff02"
EXPANDER_SOURCE_SHA = "575cf80551ca09b681bc7b0e13b46f9259c5d4504f726647277fb0b828dc710e"
FIXTURE_SOURCE_SHA = "e20285669a87285803bc6f9714b8d1ebae8188551e01a68f645ab39893e6e32c"
ACTIVE_CASE_DEVICE_FIXTURE_SHA = "d31a5240ac65a09c2f95c12fb3e54be122ba56299ee49cc39ee1d9567a5dcd73"
EXPECTED_DRIVER_SHA = "d7458fcdf8553871cac00123413676625c61eff2fdee3be9a440e656f05bcc1e"
EXPECTED_DRIVER_BYTES = 3505000
EXPECTED_DRIVER_BUILD_ID = "033ce9b214e2149861a8fcf0381c27bbac5bf1d1"
DRIVER_BUILD_METADATA = {
    "command": "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-a cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
    "provenance": "clean Git worktree at the exact source commit with an initially absent independent target directory",
    "source_commit": DRIVER_COMMIT,
    "source_tree": DRIVER_TREE,
    "source_git_blob": DRIVER_SOURCE_GIT_BLOB,
    "source_sha256": DRIVER_SOURCE_SHA,
    "cargo_version": "1.96.0 (30a34c682 2026-05-25)",
    "rustc_version": "1.96.0 (ac68faa20 2026-05-25)",
    "rustc_host": "x86_64-unknown-linux-gnu",
    "llvm_version": "22.1.2",
    "cxx_version": "c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0",
    "clang_version": "Ubuntu clang version 18.1.3 (1ubuntu1)",
    "mold_version": "mold 2.30.0 (compatible with GNU ld)",
    "rocm_resolved_path": "/opt/rocm-7.2.1",
    "cargo_build_jobs": 1,
    "cargo_incremental": False,
    "locked": True,
    "profile": "release",
    "binary_bytes": EXPECTED_DRIVER_BYTES,
    "binary_build_id_sha1": EXPECTED_DRIVER_BUILD_ID,
    "expected_binary_sha256": EXPECTED_DRIVER_SHA,
    "build_inputs": DRIVER_BUILD_INPUTS,
    "reproducibility": {
        "build_count": 2,
        "commands": [
            "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-a cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
            "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-b cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
        ],
        "independent_initially_absent_target_directories": True,
        "byte_identical": True,
        "sha256_equal": True,
        "bytes_equal": True,
        "build_id_equal": True,
    },
}
EXPECTED_SERVED_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_WORKER_SHA = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_PACKAGE_MANIFEST_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_PACKAGE_CONTENT_SHA = "a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1"
EXPECTED_CASE_MANIFEST_SHA = "1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea"
EXPECTED_GUARD_SHA = "4eafd9bc149792b9c9849fed07a70830a42cf8227b85431130eec8f41708abc0"
EXPECTED_PACKAGE_FILES = 1045
PROTOCOL = "ullm.aq4_p2_resident_driver.v2"
BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v4"
MAX_JSON = 64 * 1024 * 1024
CHUNK = 1024 * 1024
MAX_WORKER_RELEASE_ENTRIES = 4096
MAX_WORKER_RELEASE_DEPTH = 8
EXPECTED_WORKER_HARDLINK_FIXTURE_SHA = "4a6bfe06d2ebd7bbabc1ad4e4c6df24b14a1c4837466c2a675953cb1d226d340"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FD_MAP_SCHEMA = "ullm.aq4_p3_inherited_fd_map.v1"
FD_MAP_ENV = "ULLM_AQ4_PINNED_FD_MAP"
FD_MAP_MAX_BYTES = 1024 * 1024
FD_CLOSURE_CONTRACT = {
    "code_execution_closure": "pinned_fd",
    "control_input_closure": "pinned_fd",
    "device_lock_closure": "pinned_fd",
    "data_integrity": "trusted_pre_post_guarded",
}

REQUIRED_FILES = {
    "official-case.json": (0o444, "trusted_official_expansion_case"),
    "case-binding.json": (0o444, "runtime_bound_case"),
    "fixture.json": (0o444, "fixture"),
    "fixture-index.json": (0o444, "fixture_index"),
    "identity.json": (0o444, "resident_identity"),
    "preflight.json": (0o444, "synthetic_preflight"),
    "policy.json": (0o444, "threshold_policy"),
    "served-model.json": (0o444, "served_model_snapshot"),
    "package-manifest.json": (0o444, "package_manifest_snapshot"),
    "trust-roots.json": (0o444, "independent_trust_roots"),
    "launch-command.json": (0o444, "exact_resident_launch_command"),
    "trusted-runner.py": (0o444, "trusted_one_case_smoke_runner"),
    "SUPERSEDED-0fd7993.json": (0o444, "historical_non_executable_bundle_record"),
    "resident-driver": (0o555, "detached_resident_driver"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
    "runner-dry-run-evidence.json": (0o444, "trusted_runner_subprocess_evidence"),
}
POST_RUN_FILES = {"dry-run.json", "runner-dry-run-evidence.json"}
RUNNER_VALIDATE_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-validate-only")
BINDING_RUNNER_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-binding-v7-runner")
BINDING_SOURCE_COMMIT = "76c48aa27c08f8cd5115a15e6be25b83d679d8fa"
BINDING_SOURCE_TREE = "e79865753bcbba1a9134670fa2ea57327ab84ea4"
BINDING_RUNNER_GIT_BLOB = "1929ca23d50c85d3464f9a2c87f1e062d0dc665a"
BINDING_RUNNER_SHA = "bbe978ede0e4662c33d0d12eee4194531f340b9c06001f37d619019197fd5138"
BINDING_DRIVER_GIT_BLOB = "7e37119cc8b66dc0e0f7abcf49b896fcdad8315f"
BUNDLE_ROOT_MODE = 0o555
BINDING_ROOT_MODE = 0o555
BINDING_PREDECESSOR_COMMIT = "31eb65a644eae20a3be6cbeb36b04aaaabf69429"
BINDING_FAKE_READY_SCOPE = {
    "stage": "pre_spawn_fixture_only",
    "runtime_proof": False,
    "ready_proof": False,
    "model_load_proof": False,
}
BINDING_FILES = {
    "trusted-runner.py": (0o444, "actual_generic_runner_source"),
    "trusted-validator.py": (0o444, "trusted_bundle_validator_source"),
    "runner-plan.json": (0o444, "actual_generic_runner_dry_run_plan"),
    "runner-subprocess-evidence.json": (0o444, "actual_runner_subprocess_evidence"),
    "validator-report.json": (0o444, "trusted_validator_report"),
    "binding-manifest.json": (0o444, "immutable_binding_manifest"),
}
_VALIDATION_HOOK: Callable[[Path], None] | None = None
_BINDING_VALIDATION_HOOK: Callable[[Path], None] | None = None


class BundleError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=strict_pairs, parse_constant=lambda item: (_ for _ in ()).throw(BundleError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} root must be an object")
    return value


def fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def named_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


class PinnedFdMap:
    def __init__(self, descriptor: int, value: dict[str, Any]) -> None:
        self.descriptor = descriptor
        self.value = value
        self.bindings = {item["logical_path"]: item for item in value["bindings"]}
        self.roles = {item["role"]: item for item in value["bindings"]}

    @classmethod
    def from_environment(cls) -> "PinnedFdMap | None":
        raw_descriptor = os.environ.get(FD_MAP_ENV)
        if raw_descriptor is None:
            return None
        if not raw_descriptor.isascii() or not raw_descriptor.isdecimal():
            raise BundleError("pinned FD map descriptor is invalid")
        descriptor = int(raw_descriptor)
        if descriptor < 3:
            raise BundleError("pinned FD map descriptor is reserved")
        required_seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        try:
            if (
                not required_seals
                or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals
                != required_seals
            ):
                raise BundleError("pinned FD map seals differ")
            data = os.pread(descriptor, FD_MAP_MAX_BYTES + 1, 0)
        except OSError as error:
            raise BundleError(f"pinned FD map read failed: {error}") from error
        if len(data) > FD_MAP_MAX_BYTES or not data.endswith(b"\n"):
            raise BundleError("pinned FD map byte contract differs")
        value = parse_json(data[:-1], "pinned FD map")
        if set(value) != {
            "schema_version", "status", "map_sha256", "logical_argv_sha256",
            "closure_contract", "bindings",
        }:
            raise BundleError("pinned FD map root fields differ")
        if value.get("schema_version") != FD_MAP_SCHEMA or value.get("status") != "bound":
            raise BundleError("pinned FD map schema/status differs")
        declared = value.get("map_sha256")
        unhashed = dict(value)
        unhashed["map_sha256"] = None
        if (
            not isinstance(declared, str)
            or SHA_RE.fullmatch(declared) is None
            or sha_bytes(canonical(unhashed)) != declared
            or data != canonical(value) + b"\n"
        ):
            raise BundleError("pinned FD map self-hash/canonical bytes differ")
        if (
            not isinstance(value.get("logical_argv_sha256"), str)
            or SHA_RE.fullmatch(value["logical_argv_sha256"]) is None
            or value.get("closure_contract") != FD_CLOSURE_CONTRACT
        ):
            raise BundleError("pinned FD map logical argv/closure contract differs")
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise BundleError("pinned FD map bindings are missing")
        paths: set[str] = set()
        roles: set[str] = set()
        descriptors: set[int] = {descriptor}
        allowed = {
            "code_execution": {"exec", "dlopen"},
            "control_input": {"read"},
            "device_lock": {"flock"},
            "data_integrity": {"pre_post_guard"},
        }
        for item in bindings:
            if not isinstance(item, dict) or set(item) != {
                "role", "logical_path", "resolved_path", "descriptor", "kind",
                "closure", "method", "identity", "sha256",
            }:
                raise BundleError("pinned FD map binding fields differ")
            role = item.get("role")
            logical_path = item.get("logical_path")
            resolved_path = item.get("resolved_path")
            child_descriptor = item.get("descriptor")
            kind = item.get("kind")
            closure = item.get("closure")
            method = item.get("method")
            identity = item.get("identity")
            if (
                not isinstance(role, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{1,63}", role) is None
                or role in roles
                or not isinstance(logical_path, str)
                or not Path(logical_path).is_absolute()
                or ".." in Path(logical_path).parts
                or logical_path in paths
                or type(child_descriptor) is not int
                or child_descriptor < 3
                or child_descriptor in descriptors
                or kind not in {"regular_file", "directory", "symlinked_file"}
                or closure not in allowed
                or method not in allowed[closure]
                or (kind == "directory") != (method == "pre_post_guard")
                or (
                    kind == "symlinked_file"
                    and (
                        not isinstance(resolved_path, str)
                        or not Path(resolved_path).is_absolute()
                        or ".." in Path(resolved_path).parts
                    )
                )
                or (kind != "symlinked_file" and resolved_path is not None)
                or not isinstance(identity, dict)
                or set(identity) != {
                    "device", "inode", "mode", "nlink", "size", "mtime_ns", "ctime_ns",
                }
                or any(type(part) is not int for part in identity.values())
                or (
                    method in {"exec", "dlopen", "read"}
                    and (
                        not isinstance(item.get("sha256"), str)
                        or SHA_RE.fullmatch(item["sha256"]) is None
                    )
                )
                or (method in {"flock", "pre_post_guard"} and item.get("sha256") is not None)
            ):
                raise BundleError("pinned FD map binding value differs")
            try:
                metadata = os.fstat(child_descriptor)
            except OSError as error:
                raise BundleError(f"pinned FD binding is unavailable: {role}: {error}") from error
            if kind == "directory":
                correct_kind = stat.S_ISDIR(metadata.st_mode)
            else:
                correct_kind = stat.S_ISREG(metadata.st_mode)
            if not correct_kind or not cls._identity_matches(item, metadata):
                raise BundleError(f"pinned FD binding identity/type differs: {role}")
            paths.add(logical_path)
            roles.add(role)
            descriptors.add(child_descriptor)
        result = cls(descriptor, value)
        for item in bindings:
            if item["method"] in {"exec", "dlopen", "read"}:
                result.read(item, collect=False)
        result.verify_data_guards()
        return result

    @staticmethod
    def _identity_matches(item: dict[str, Any], metadata: os.stat_result) -> bool:
        observed = named_identity(metadata)
        if item["method"] == "flock":
            keys = {"device", "inode", "mode", "nlink"}
        elif item["method"] == "pre_post_guard":
            keys = set(observed)
        else:
            keys = {"device", "inode", "mode", "nlink", "size", "mtime_ns"}
        return all(observed[key] == item["identity"][key] for key in keys)

    def binding(self, path: Path) -> dict[str, Any] | None:
        return self.bindings.get(str(path))

    def role(self, role: str) -> dict[str, Any]:
        item = self.roles.get(role)
        if item is None:
            raise BundleError(f"required pinned FD role is missing: {role}")
        return item

    def verify_binding(self, item: dict[str, Any]) -> os.stat_result:
        try:
            metadata = os.fstat(item["descriptor"])
        except OSError as error:
            raise BundleError(f"pinned FD binding disappeared: {item['role']}: {error}") from error
        if not self._identity_matches(item, metadata):
            raise BundleError(f"pinned FD binding identity changed: {item['role']}")
        return metadata

    def read(
        self,
        item: dict[str, Any],
        *,
        maximum: int | None = None,
        collect: bool = True,
    ) -> tuple[bytes, str, os.stat_result]:
        if item["method"] not in {"exec", "dlopen", "read"}:
            raise BundleError(f"pinned FD binding is not readable: {item['role']}")
        before = self.verify_binding(item)
        if maximum is not None and before.st_size > maximum:
            raise BundleError(f"pinned FD binding exceeds bounded size: {item['role']}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(item["descriptor"], min(CHUNK, before.st_size - offset), offset)
            if not chunk:
                raise BundleError(f"pinned FD binding ended early: {item['role']}")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
            offset += len(chunk)
        after = self.verify_binding(item)
        observed_sha = digest.hexdigest()
        if not self._identity_matches(item, after) or observed_sha != item["sha256"]:
            raise BundleError(f"pinned FD binding bytes changed: {item['role']}")
        return b"".join(chunks), observed_sha, before

    def verify_data_guards(self) -> None:
        for item in self.value["bindings"]:
            metadata = self.verify_binding(item)
            if item["method"] != "pre_post_guard":
                continue
            path = Path(item["logical_path"])
            try:
                logical = os.lstat(path)
            except OSError as error:
                raise BundleError(f"guarded data path is unavailable: {path}: {error}") from error
            if named_identity(logical) != item["identity"] or named_identity(metadata) != item["identity"]:
                raise BundleError(f"guarded data path identity changed: {path}")


def reject_symlink_components(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise BundleError(f"{label} path must be absolute without parent traversal")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise BundleError(f"{label} path has a symlink component: {current}")
        except FileNotFoundError:
            continue


class Snapshot:
    def __init__(self) -> None:
        self.entries: dict[Path, tuple[int, ...]] = {}

    def capture(self, path: Path, value: os.stat_result | None = None) -> os.stat_result:
        reject_symlink_components(path, "trust root")
        observed = path.lstat() if value is None else value
        prior = self.entries.get(path)
        if prior is not None and prior != fingerprint(observed):
            raise BundleError(f"trust root changed during reconstruction: {path}")
        self.entries[path] = fingerprint(observed)
        return observed

    def verify(self) -> None:
        for path, expected in self.entries.items():
            if fingerprint(path.lstat()) != expected:
                raise BundleError(f"trust-root TOCTOU mutation detected: {path}")


def read_stable(path: Path, label: str, maximum: int | None = MAX_JSON, *, single_link: bool = True, snapshot: Snapshot | None = None) -> bytes:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    if maximum is not None and before.st_size > maximum:
        raise BundleError(f"{label} exceeds bounded size")
    if snapshot is not None:
        snapshot.capture(path, before)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        if fingerprint(os.fstat(descriptor)) != fingerprint(before):
            raise BundleError(f"{label} changed before open")
        while chunk := os.read(descriptor, CHUNK):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if fingerprint(after) != fingerprint(before) or fingerprint(path.lstat()) != fingerprint(before):
        raise BundleError(f"{label} changed while read")
    return b"".join(chunks)


def sha_file(path: Path, label: str, *, single_link: bool = True, snapshot: Snapshot | None = None) -> str:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    if snapshot is not None:
        snapshot.capture(path, before)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        if fingerprint(os.fstat(descriptor)) != fingerprint(before):
            raise BundleError(f"{label} changed before hash")
        while chunk := os.read(descriptor, CHUNK):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if fingerprint(after) != fingerprint(before) or fingerprint(path.lstat()) != fingerprint(before):
        raise BundleError(f"{label} changed while hashed")
    return digest.hexdigest()


def worker_stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": value.st_mode,
        "size": value.st_size,
        "nlink": value.st_nlink,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def scan_worker_inode_paths(root: Path, expected: dict[str, int], label: str, snapshot: Snapshot | None = None) -> set[Path]:
    reject_symlink_components(root, label)
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise BundleError(f"{label} must be a no-symlink directory")
    if snapshot is not None:
        snapshot.capture(root, root_metadata)
    pending = [(root, 0)]
    visited = 0
    matches: set[Path] = set()
    while pending:
        directory, depth = pending.pop()
        if depth > MAX_WORKER_RELEASE_DEPTH:
            raise BundleError(f"{label} depth exceeds bound")
        for entry in os.scandir(directory):
            path = Path(entry.path)
            visited += 1
            if visited > MAX_WORKER_RELEASE_ENTRIES:
                raise BundleError(f"{label} entry count exceeds bound")
            if entry.is_symlink():
                raise BundleError(f"{label} symlink rejected: {path}")
            metadata = path.lstat()
            if entry.is_dir(follow_symlinks=False):
                pending.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                if metadata.st_dev == expected["device"] and metadata.st_ino == expected["inode"]:
                    matches.add(path)
            else:
                raise BundleError(f"{label} non-regular entry rejected: {path}")
    return matches


def validate_worker_hardlink_fixture(snapshot: Snapshot | None = None, hook: Callable[[], None] | None = None, *, fixture_path: Path = WORKER_HARDLINK_FIXTURE_PATH, expected_fixture_sha: str = EXPECTED_WORKER_HARDLINK_FIXTURE_SHA) -> dict[str, Any]:
    fixture_raw = read_stable(fixture_path, "worker link fixture", snapshot=snapshot)
    if sha_bytes(fixture_raw) != expected_fixture_sha:
        raise BundleError("worker link fixture SHA differs")
    fixture = parse_json(fixture_raw, "worker link fixture")
    exact = {"schema_version", "roots", "paths", "primary_path", "sha256", "expected"}
    if set(fixture) != exact or fixture.get("schema_version") != "ullm.aq4_p2_resident_worker_link_identity.v2" or not isinstance(fixture.get("sha256"), str) or SHA_RE.fullmatch(fixture["sha256"]) is None:
        raise BundleError("worker link fixture schema differs")
    expected = fixture.get("expected")
    expected_keys = {"device", "inode", "uid", "gid", "mode", "size", "nlink", "mtime_ns", "ctime_ns"}
    roots_raw = fixture.get("roots")
    paths_raw = fixture.get("paths")
    if not isinstance(expected, dict) or set(expected) != expected_keys or any(type(expected[key]) is not int or expected[key] < 0 for key in expected_keys) or not isinstance(roots_raw, list) or not roots_raw or not isinstance(paths_raw, list) or not paths_raw or any(not isinstance(item, str) for item in [*roots_raw, *paths_raw]) or expected["nlink"] != len(paths_raw) or expected["nlink"] == 0:
        raise BundleError("worker link fixture metadata differs")
    roots = [Path(item) for item in roots_raw]
    paths = [Path(item) for item in paths_raw]
    primary = Path(fixture["primary_path"])
    if len(set(roots)) != len(roots) or len(set(paths)) != len(paths) or primary != paths[0]:
        raise BundleError("worker link paths/roots differ")
    for path in roots:
        reject_symlink_components(path, "worker scan root")
    for path in paths:
        reject_symlink_components(path, "worker declared path")
        if not any(path != root and path.is_relative_to(root) for root in roots):
            raise BundleError("worker link declared path is outside scan roots")
    try:
        before = [path.lstat() for path in paths]
    except OSError as error:
        raise BundleError(f"worker link pre-open metadata failed: {error}") from error
    if any(not stat.S_ISREG(item.st_mode) or worker_stat_identity(item) != expected or item.st_mode & 0o111 == 0 or item.st_mode & 0o002 for item in before):
        raise BundleError("worker link pre-open metadata differs")
    if snapshot is not None:
        for path, metadata in zip(paths, before, strict=True):
            snapshot.capture(path, metadata)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        for path in paths:
            descriptors.append(os.open(path, flags))
        if any(worker_stat_identity(os.fstat(descriptor)) != expected for descriptor in descriptors):
            raise BundleError("worker link open metadata differs")
        for root in roots:
            expected_paths = {path for path in paths if path.is_relative_to(root)}
            if scan_worker_inode_paths(root, expected, "worker scan root", snapshot) != expected_paths:
                raise BundleError("worker link exact path coverage differs")
        for descriptor in descriptors:
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, CHUNK):
                digest.update(chunk)
                size += len(chunk)
            if digest.hexdigest() != fixture["sha256"] or size != expected["size"]:
                raise BundleError("worker link FD hash differs")
        if hook is not None:
            hook()
        try:
            post = [path.lstat() for path in paths]
        except OSError as error:
            raise BundleError(f"worker link post metadata failed: {error}") from error
        if any(worker_stat_identity(item) != expected for item in post) or any(worker_stat_identity(os.fstat(descriptor)) != expected for descriptor in descriptors):
            raise BundleError("worker link identity changed during validation")
        for root in roots:
            expected_paths = {path for path in paths if path.is_relative_to(root)}
            if scan_worker_inode_paths(root, expected, "worker scan root post", snapshot) != expected_paths:
                raise BundleError("worker link post path coverage differs")
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return {
        "fixture": {"path": str(fixture_path), "sha256": expected_fixture_sha},
        "roots": [str(path) for path in roots],
        "paths": [str(path) for path in paths],
        "primary_path": str(primary),
        "sha256": fixture["sha256"],
        "expected": expected,
        "exact_path_count": len(paths),
        "unknown_hardlinks_possible": False,
    }


def package_tree_sha256(root: Path, snapshot: Snapshot | None = None) -> tuple[str, int]:
    reject_symlink_components(root, "package root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("package root must be a non-symlink directory")
    pending = [(root, 0)]
    files: list[Path] = []
    while pending:
        directory, depth = pending.pop()
        if depth > 32:
            raise BundleError("package depth exceeds 32")
        if snapshot is not None:
            snapshot.capture(directory)
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_symlink():
                raise BundleError(f"package symlink rejected: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
                if len(files) > 65_536:
                    raise BundleError("package file count exceeds 65536")
            else:
                raise BundleError(f"package non-regular entry rejected: {path}")
    if not files:
        raise BundleError("package tree is empty")
    aggregate = hashlib.sha256()
    for path in sorted(files, key=str):
        relative = path.relative_to(root).as_posix()
        digest = bytes.fromhex(sha_file(path, f"package/{relative}", snapshot=snapshot))
        aggregate.update(relative.encode() + b"\0" + digest + b"\n")
    return aggregate.hexdigest(), len(files)


def git_blob(path: str, expected_sha: str, commit: str = SOURCE_COMMIT) -> bytes:
    completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or sha_bytes(completed.stdout) != expected_sha:
        raise BundleError(f"trusted Git blob differs: {path}")
    return completed.stdout


def case_hash(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return sha_bytes(canonical(clone))


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value))
    clone[field] = None
    return sha_bytes(canonical(clone))


def guard_sha(names: list[str]) -> str:
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    for name in sorted(names):
        digest.update(f"{name}=1\n".encode())
    return digest.hexdigest()


def token_ids(case: dict[str, Any], count: int, vocab_size: int, reserved: set[int]) -> list[int]:
    result: list[int] = []
    index = 0
    seed = f"{case['case_id']}\0{case['case_sha256']}".encode()
    while len(result) < count:
        digest = hashlib.sha256(seed + index.to_bytes(8, "little")).digest()
        index += 1
        candidate = int.from_bytes(digest[:8], "little") % vocab_size
        if candidate not in reserved:
            result.append(candidate)
    return result


def runner_validate_argv() -> list[str]:
    return [
        str(Path(sys.executable).resolve()), str(CANONICAL_ROOT / "trusted-runner.py"),
        "--expanded", str(CANONICAL_ROOT / "case-binding.json"),
        "--fixture-index", str(CANONICAL_ROOT / "fixture-index.json"),
        "--identity", str(CANONICAL_ROOT / "identity.json"),
        "--preflight", str(CANONICAL_ROOT / "preflight.json"),
        "--policy", str(CANONICAL_ROOT / "policy.json"),
        "--output-dir", str(RUNNER_VALIDATE_OUTPUT),
        "--run-id", "p2-r9700-resident-one-case-smoke-runner-validate-v4",
        "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run",
    ]


def resident_driver_argv() -> list[str]:
    return [str(CANONICAL_ROOT / "resident-driver"), "--served-model-manifest", str(SERVED_PATH), "--device-index", "1", "--build-git-commit", DRIVER_COMMIT]


def trusted_official_case(manifest: dict[str, Any], expander_source: bytes) -> dict[str, Any]:
    module = types.ModuleType("trusted_aq4_p2_expander")
    exec(compile(expander_source, f"{SOURCE_COMMIT}:tools/expand-aq4-production-p2.py", "exec"), module.__dict__)
    expanded = module.expand(json.loads(json.dumps(manifest)), EXPECTED_CASE_MANIFEST_SHA)
    matches = [case for case in expanded["cases"] if case.get("stage_id") == "representative" and case.get("scope") == "full_model" and case.get("phase") == "cold_prefill" and case.get("mode") == "cold_batched" and case.get("prompt_tokens") == 128 and case.get("prefill_requested_m") == 128 and case.get("device", {}).get("device_id") == "r9700-rdna4" and case.get("control_id") == "aq4_0_target"]
    if len(matches) != 1 or matches[0].get("case_sha256") != case_hash(matches[0]):
        raise BundleError("trusted official expansion case differs")
    return matches[0]


def bind_runtime_case(source: dict[str, Any]) -> dict[str, Any]:
    case = json.loads(json.dumps(source))
    case["device"] = {"device_id": "r9700-rdna4", "runtime_device_index": 1, "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"}
    case["case_sha256"] = case_hash(case)
    return case


class Reconstruction:
    def __init__(self, payloads: dict[str, bytes], bundle: dict[str, Any], sums: bytes, snapshot: Snapshot) -> None:
        self.payloads = payloads
        self.bundle = bundle
        self.sums = sums
        self.snapshot = snapshot


def reconstruct() -> Reconstruction:
    snapshot = Snapshot()
    tree = subprocess.run(["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if tree.returncode != 0 or tree.stdout.strip() != SOURCE_TREE:
        raise BundleError("trusted source commit/tree differs")
    driver_tree = subprocess.run(["git", "rev-parse", f"{DRIVER_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if driver_tree.returncode != 0 or driver_tree.stdout.strip() != DRIVER_TREE:
        raise BundleError("normative driver commit/tree differs")
    driver_object = subprocess.run(
        ["git", "rev-parse", f"{DRIVER_COMMIT}:crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if driver_object.returncode != 0 or driver_object.stdout.strip() != DRIVER_SOURCE_GIT_BLOB:
        raise BundleError("normative driver Git blob differs")
    for path, authority in DRIVER_BUILD_INPUTS.items():
        observed = subprocess.run(
            ["git", "rev-parse", f"{DRIVER_COMMIT}:{path}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if observed.returncode != 0 or observed.stdout.strip() != authority["git_blob"]:
            raise BundleError(f"normative driver build input Git blob differs: {path}")
        git_blob(path, authority["sha256"], DRIVER_COMMIT)
    runner_tree = subprocess.run(["git", "rev-parse", f"{RUNNER_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if runner_tree.returncode != 0 or runner_tree.stdout.strip() != RUNNER_TREE:
        raise BundleError("prepared runner commit/tree differs")
    driver_source = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA)
    normative_driver_source = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, DRIVER_COMMIT)
    if driver_source != normative_driver_source:
        raise BundleError("current source changed the normative resident driver blob")
    if f'const PROTOCOL: &str = "{PROTOCOL}";'.encode() not in driver_source:
        raise BundleError("trusted protocol differs")
    for contract in (
        b'require_absolute_normal_path(&args.served_model_manifest, "served model manifest")?',
        b'require_absolute_normal_path(path, label)?',
        b'Component::ParentDir',
        b'metadata.nlink() != 1',
        b'WorkerHardlinkGuard::capture(&model.worker.binary, &model.worker.binary_sha256)?',
        b'worker_guard.verify(&model.worker.binary, &model.worker.binary_sha256)?',
        b'case.device.architecture != identity.runtime_device.architecture',
        b'fixture.paths.len() as u64 != fixture.expected.nlink',
        b'worker link path coverage differs',
        b'MAX_WORKER_RELEASE_ENTRIES',
        b'O_NOFOLLOW',
    ):
        if contract not in driver_source:
            raise BundleError("trusted general-single-link/worker-exact-two protocol contract differs")
    runner_source = git_blob("tools/run-aq4-p2-resident-batch.py", RUNNER_SOURCE_SHA, RUNNER_COMMIT)
    for contract in (b"def validate_driver_command", b"expected_binary_sha256", b"LOCK_EX | fcntl.LOCK_NB"):
        if contract not in runner_source:
            raise BundleError("trusted runner launch contract differs")
    expander_source = git_blob("tools/expand-aq4-production-p2.py", EXPANDER_SOURCE_SHA)
    git_blob("tools/generate-aq4-p2-fixtures.py", FIXTURE_SOURCE_SHA)
    active_case_device_fixture_raw = git_blob(
        "tests/fixtures/aq4-p2-resident-case-device/active-production.json",
        ACTIVE_CASE_DEVICE_FIXTURE_SHA,
    )
    active_case_device_fixture = parse_json(active_case_device_fixture_raw, "active case device fixture")
    fixture_case_device = active_case_device_fixture.get("case", {}).get("device", {})
    fixture_runtime_device = active_case_device_fixture.get("runtime_device", {})
    fixture_source_device = active_case_device_fixture.get("source_device", {})
    if (
        active_case_device_fixture.get("schema_version") != "ullm.aq4_p2_resident_case_device_identity.v1"
        or fixture_case_device != fixture_runtime_device
        or fixture_case_device.get("architecture") != "gfx1201"
        or fixture_source_device.get("architecture") != "RDNA4"
        or fixture_case_device.get("backend") != "hip"
        or fixture_source_device.get("backend") != "hip"
    ):
        raise BundleError("active case source/runtime device vocabulary differs")

    served_raw = read_stable(SERVED_PATH, "active served model", snapshot=snapshot)
    if sha_bytes(served_raw) != EXPECTED_SERVED_SHA:
        raise BundleError("active served model trust-root SHA differs")
    served = parse_json(served_raw, "active served model")
    public, generation, worker, product, format_value = (served.get(key) for key in ("public", "generation", "worker", "product", "format"))
    if served.get("schema_version") != "ullm.served_model.v2" or not all(isinstance(value, dict) for value in (public, generation, worker, product, format_value)):
        raise BundleError("active served model nested schema differs")
    if public.get("id") != "ullm-qwen3.5-9b-aq4" or public.get("revision") != "aq4-reasoning-v0.1-candidate" or worker.get("protocol") != "ullm.worker.v2" or worker.get("identity") != {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"} or format_value != {"format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1"}:
        raise BundleError("active served model semantic trust root differs")
    worker_path = Path(worker.get("binary", ""))
    if worker_path != ROOT / "target/reasoning-v2/release/ullm-aq4-worker":
        raise BundleError("active worker path differs")
    worker_hardlinks = validate_worker_hardlink_fixture(snapshot)
    if worker_hardlinks["paths"][0] != str(worker_path) or worker_hardlinks["sha256"] != EXPECTED_WORKER_SHA or worker.get("binary_sha256") != EXPECTED_WORKER_SHA:
        raise BundleError("active worker identity differs")
    guards = worker.get("required_environment")
    if not isinstance(guards, list) or len(guards) != len(set(guards)) or any(not isinstance(item, str) for item in guards) or guard_sha(guards) != EXPECTED_GUARD_SHA:
        raise BundleError("active guard trust root differs")

    package_info = product.get("package")
    if not isinstance(package_info, dict) or product.get("root") != "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1":
        raise BundleError("active product root differs")
    package_manifest_path = Path(product["root"]) / package_info.get("manifest_path", "")
    package_manifest_raw = read_stable(package_manifest_path, "package manifest", snapshot=snapshot)
    if sha_bytes(package_manifest_raw) != EXPECTED_PACKAGE_MANIFEST_SHA or package_info.get("manifest_sha256") != EXPECTED_PACKAGE_MANIFEST_SHA:
        raise BundleError("package manifest trust root differs")
    parse_json(package_manifest_raw, "package manifest")
    package_sha, package_count = package_tree_sha256(package_manifest_path.parent, snapshot)
    if package_sha != EXPECTED_PACKAGE_CONTENT_SHA or package_count != EXPECTED_PACKAGE_FILES:
        raise BundleError("package content trust root differs")

    manifest_raw = read_stable(CASE_MANIFEST_PATH, "official case manifest", snapshot=snapshot)
    if sha_bytes(manifest_raw) != EXPECTED_CASE_MANIFEST_SHA:
        raise BundleError("official case manifest trust root differs")
    source_case = trusted_official_case(parse_json(manifest_raw, "official case manifest"), expander_source)
    case = bind_runtime_case(source_case)
    runtime_device = case["device"]
    source_case_raw = pretty({"schema_version": "ullm.aq4_p2_official_case.v1", "source_commit": SOURCE_COMMIT, "manifest_sha256": EXPECTED_CASE_MANIFEST_SHA, "case": source_case})
    case_binding = {
        "schema_version": "ullm.aq4_production_p2_expanded.v2",
        "status": "bound_one_case_smoke",
        "source_manifest_sha256": EXPECTED_CASE_MANIFEST_SHA,
        "official_case_sha256": source_case["case_sha256"],
        "runtime_binding": {"schema_version": "ullm.aq4_p2_r9700_host_binding.v1", "source_device": source_case["device"], "bound_device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}},
        "case_count": 1,
        "canonical_case_sha256": sha_bytes(canonical([case])),
        "cases": [case],
    }
    case_binding_raw = pretty(case_binding)
    case_binding_sha = sha_bytes(case_binding_raw)
    reasoning = served.get("reasoning")
    if not isinstance(reasoning, dict):
        raise BundleError("served reasoning contract differs")
    reserved = set(generation.get("eos_token_ids", []))
    for field in ("start_token_ids", "end_token_ids", "forced_end_token_ids"):
        values = reasoning.get(field)
        if not isinstance(values, list):
            raise BundleError(f"served reasoning {field} differs")
        reserved.update(values)
    ids = token_ids(case, case["prompt_tokens"], generation["vocab_size"], reserved)
    fixture = {"schema_version": "ullm.aq4_p2_case_fixture.v1", "cases": [{"case_id": case["case_id"], "prompt_token_ids": ids, "step_count": case["generated_tokens"]}]}
    fixture_raw = pretty(fixture)
    fixture_sha = sha_bytes(fixture_raw)
    fixture_index = {
        "schema_version": "ullm.aq4_p2_fixture_index.v1",
        "expanded_manifest_sha256": case_binding_sha,
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "subset": "resident_one_case_smoke",
        "case_count": 1,
        "cases": [{"case_id": case["case_id"], "case_sha256": case["case_sha256"], "fixture_path": str(CANONICAL_ROOT / "fixture.json"), "fixture_sha256": fixture_sha, "prompt_tokens": case["prompt_tokens"], "context_tokens": case["context_tokens"], "generated_tokens": case["generated_tokens"], "prompt_token_ids_sha256": sha_bytes(canonical(ids))}],
    }
    fixture_index_raw = pretty(fixture_index)
    resident_identity = {
        "binary_sha256": EXPECTED_DRIVER_SHA, "build_git_commit": DRIVER_COMMIT, "protocol": PROTOCOL,
        "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA,
        "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "model_id": public["id"], "model_revision": public["revision"], "format_id": format_value["format_id"],
        "implementation_id": format_value["implementation_id"], "runtime_device": runtime_device, "guard_set_sha256": EXPECTED_GUARD_SHA,
    }
    identity = {
        "schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None,
        "expanded_manifest_sha256": case_binding_sha, "build_git_commit": DRIVER_COMMIT, "resident_driver_identity": resident_identity,
        "hash_binding": {"bound_case_manifest_sha256": case_binding_sha, "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA, "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "served_model_manifest_sha256": EXPECTED_SERVED_SHA},
    }
    identity["identity_sha256"] = self_hash(identity, "identity_sha256")
    identity_raw = pretty(identity)
    preflight = {field: 0 for field in ("weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes")}
    preflight["gpu_process_snapshot"] = []
    policy = {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}
    fake_ready = {"event": "ready", "schema_version": PROTOCOL, "model_loads": 1, "resident_session_id": "offline-fake-ready-not-executed", "driver_identity": resident_identity}
    trust_roots = {
        "schema_version": "ullm.aq4_p2_resident_smoke_trust_roots.v1",
        "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "driver_source_git_blob": DRIVER_SOURCE_GIT_BLOB, "driver_source_sha256": DRIVER_SOURCE_SHA, "driver_build_inputs": DRIVER_BUILD_INPUTS, "runner_source_commit": RUNNER_COMMIT, "runner_source_tree": RUNNER_TREE, "runner_source_sha256": RUNNER_SOURCE_SHA, "expander_source_sha256": EXPANDER_SOURCE_SHA, "fixture_generator_source_sha256": FIXTURE_SOURCE_SHA, "active_case_device_fixture_sha256": ACTIVE_CASE_DEVICE_FIXTURE_SHA, "protocol": PROTOCOL, "one_case_smoke_runner": {"flag": "--one-case-smoke", "normal_case_count": 84, "smoke_case_count": 1, "smoke_transactions": 12, "warmup_runs": 2, "measured_runs": 10, "promotion_eligible": False}, "normative_driver": {"commit": DRIVER_COMMIT, "tree": DRIVER_TREE, "git_blob": DRIVER_SOURCE_GIT_BLOB, "source_sha256": DRIVER_SOURCE_SHA, "build_inputs": DRIVER_BUILD_INPUTS, "blob_unchanged_at_current_source": True, "clean_build": DRIVER_BUILD_METADATA}, "protocol_path_contract": {"served_model_manifest": "absolute_without_parent_traversal", "case_binding": "absolute_without_parent_traversal", "identity": "absolute_without_parent_traversal", "preflight": "absolute_without_parent_traversal", "policy": "absolute_without_parent_traversal", "fixture": "absolute_without_parent_traversal"}},
        "external": {"served_model": {"path": str(SERVED_PATH), "sha256": EXPECTED_SERVED_SHA}, "worker": {"path": str(worker_path), "sha256": EXPECTED_WORKER_SHA, "hardlink_set": worker_hardlinks}, "package_manifest": {"path": str(package_manifest_path), "sha256": EXPECTED_PACKAGE_MANIFEST_SHA}, "package_tree": {"path": str(package_manifest_path.parent), "sha256": EXPECTED_PACKAGE_CONTENT_SHA, "file_count": EXPECTED_PACKAGE_FILES}, "case_manifest": {"path": str(CASE_MANIFEST_PATH), "sha256": EXPECTED_CASE_MANIFEST_SHA}, "guard_set_sha256": EXPECTED_GUARD_SHA},
    }
    policy_raw = pretty(policy)
    launch_command = {
        "schema_version": "ullm.aq4_p2_resident_launch_command.v1",
        "runner_validate_only_argv": runner_validate_argv(),
        "resident_driver_argv": resident_driver_argv(),
        "bindings": {"python": {"path": str(Path(sys.executable).resolve()), "sha256": sha_file(Path(sys.executable).resolve(), "Python interpreter", single_link=False)}, "runner": {"path": str(CANONICAL_ROOT / "trusted-runner.py"), "sha256": RUNNER_SOURCE_SHA, "source_commit": RUNNER_COMMIT}, "driver": {"path": str(CANONICAL_ROOT / "resident-driver"), "sha256": EXPECTED_DRIVER_SHA, "source_commit": DRIVER_COMMIT, "source_tree": DRIVER_TREE, "source_git_blob": DRIVER_SOURCE_GIT_BLOB, "source_sha256": DRIVER_SOURCE_SHA, "build": DRIVER_BUILD_METADATA}, "served_model_manifest": {"path": str(SERVED_PATH), "sha256": EXPECTED_SERVED_SHA}, "device_index": 1, "build_git_commit": DRIVER_COMMIT, "protocol": PROTOCOL, "one_case_smoke": True},
    }
    superseded = {
        "schema_version": "ullm.aq4_p2_resident_smoke_supersession.v1",
        "source_commit": "0fd7993843d0d7f1096d89079ce06922871d9f1a",
        "resident_binary_sha256": "cb81b05e6e3b80426843be0c63aa6f2beeb3686016f64a03b6af5fe019caa2b4",
        "status": "superseded_historical_prepared",
        "promotion": False,
        "execution_eligible": False,
        "superseded_by_source_commit": DRIVER_COMMIT,
        "reason": "source predates the normative resident launch-boundary hardening",
    }
    payloads = {
        "official-case.json": source_case_raw, "case-binding.json": case_binding_raw, "fixture.json": fixture_raw,
        "fixture-index.json": fixture_index_raw, "identity.json": identity_raw, "preflight.json": pretty(preflight),
        "policy.json": policy_raw, "served-model.json": served_raw, "package-manifest.json": package_manifest_raw,
        "trust-roots.json": pretty(trust_roots), "fake-ready.json": pretty(fake_ready), "trusted-runner.py": runner_source,
        "launch-command.json": pretty(launch_command), "SUPERSEDED-0fd7993.json": pretty(superseded),
    }
    file_bindings = {name: {"sha256": EXPECTED_DRIVER_SHA if name == "resident-driver" else sha_bytes(payloads[name]), "mode": f"{mode:04o}", "role": role} for name, (mode, role) in sorted(REQUIRED_FILES.items()) if name not in POST_RUN_FILES}
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "status": "prepared_not_executed", "promotion": False,
        "run_id": "p2-r9700-resident-one-case-smoke-prepared-v4", "canonical_root": str(CANONICAL_ROOT),
        "historical_predecessor": {"source_commit": superseded["source_commit"], "status": superseded["status"], "execution_eligible": False},
        "resident_driver": {"source_commit": DRIVER_COMMIT, "source_tree": DRIVER_TREE, "source_git_blob": DRIVER_SOURCE_GIT_BLOB, "source_sha256": DRIVER_SOURCE_SHA, "build_inputs": DRIVER_BUILD_INPUTS, "blob_unchanged_at_source_commit": SOURCE_COMMIT, "binary_sha256": EXPECTED_DRIVER_SHA, "binary_bytes": EXPECTED_DRIVER_BYTES, "binary_build_id_sha1": EXPECTED_DRIVER_BUILD_ID, "build": DRIVER_BUILD_METADATA, "protocol": PROTOCOL},
        "runner": {"source_commit": RUNNER_COMMIT, "source_tree": RUNNER_TREE, "source_sha256": RUNNER_SOURCE_SHA, "one_case_smoke": True},
        "expected_runtime": {"device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}, "required_guards": {name: "1" for name in sorted(guards)}},
        "bindings": {"official_case_sha256": source_case["case_sha256"], "case_sha256": case["case_sha256"], "case_binding_sha256": case_binding_sha, "fixture_sha256": fixture_sha, "identity_file_sha256": sha_bytes(identity_raw), "identity_self_sha256": identity["identity_sha256"], "preflight_sha256": sha_bytes(payloads["preflight.json"]), "policy_sha256": sha_bytes(policy_raw), "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA, "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "guard_set_sha256": EXPECTED_GUARD_SHA},
        "offline_evidence": {"trust_root_reconstruction": "passed", "schema_hash_path_link_toctou_validation": "passed", "trusted_runner_subprocess_required": True, "runner_dry_run": "passed", "synthetic_fake_ready_validation": "passed", "model_load_executed": False, "gpu_command_executed": False, "service_touched": False},
        "actual_live_observations": {"runtime_identity": None, "power": None, "vram": None, "reason": "not acquired; preparation intentionally performed no GPU model load or live service operation"},
        "files": file_bindings,
    }
    bundle_raw = pretty(bundle)
    return Reconstruction(payloads, bundle, b"", snapshot)


def safe_member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or Path(name).parts != (name,) or name in {".", ".."}:
        raise BundleError(f"unsafe bundle member path: {name!r}")
    return root / name


_safe_member = safe_member


def validate_launch_command(value: dict[str, Any]) -> None:
    expected_keys = {"schema_version", "runner_validate_only_argv", "resident_driver_argv", "bindings"}
    if set(value) != expected_keys or value.get("schema_version") != "ullm.aq4_p2_resident_launch_command.v1":
        raise BundleError("launch command exact schema differs")
    runner_argv = runner_validate_argv()
    driver_argv = resident_driver_argv()
    if value.get("runner_validate_only_argv") != runner_argv or value.get("resident_driver_argv") != driver_argv:
        raise BundleError("launch command argv differs")
    bindings = value.get("bindings")
    expected_bindings = {
        "python": {"path": runner_argv[0], "sha256": sha_file(Path(runner_argv[0]), "Python interpreter", single_link=False)},
        "runner": {"path": runner_argv[1], "sha256": RUNNER_SOURCE_SHA, "source_commit": RUNNER_COMMIT},
        "driver": {
            "path": driver_argv[0],
            "sha256": EXPECTED_DRIVER_SHA,
            "source_commit": DRIVER_COMMIT,
            "source_tree": DRIVER_TREE,
            "source_git_blob": DRIVER_SOURCE_GIT_BLOB,
            "source_sha256": DRIVER_SOURCE_SHA,
            "build": DRIVER_BUILD_METADATA,
        },
        "served_model_manifest": {"path": driver_argv[2], "sha256": EXPECTED_SERVED_SHA},
        "device_index": 1,
        "build_git_commit": DRIVER_COMMIT,
        "protocol": PROTOCOL,
        "one_case_smoke": True,
    }
    if bindings != expected_bindings:
        raise BundleError("launch command path/SHA bindings differ")
    for path, label in ((Path(runner_argv[0]), "launch Python"), (Path(runner_argv[1]), "launch runner"), (Path(driver_argv[0]), "launch driver"), (Path(driver_argv[2]), "launch served manifest")):
        reject_symlink_components(path, label)
    if "--one-case-smoke" not in runner_argv or "--dry-run" not in runner_argv:
        raise BundleError("launch command does not select validate-only one-case smoke")
    if sha_file(Path(runner_argv[1]), "launch runner") != RUNNER_SOURCE_SHA:
        raise BundleError("launch runner path/SHA differs")
    if sha_file(Path(driver_argv[0]), "launch driver") != EXPECTED_DRIVER_SHA:
        raise BundleError("launch driver path/SHA differs")
    if sha_file(Path(driver_argv[2]), "launch served manifest") != EXPECTED_SERVED_SHA:
        raise BundleError("launch served manifest path/SHA differs")


def validate_runner_plan(raw: bytes, bundle_raw: bytes, expected: Reconstruction) -> dict[str, Any]:
    plan = parse_json(raw, "trusted runner dry-run plan")
    exact_keys = {
        "schema_version", "status", "scope", "case_count", "warmup_runs", "measured_runs",
        "transaction_count", "prompt_tokens_across_transactions", "resident_model_loads",
        "baseline_identity", "links", "execution_mode", "smoke_only", "promotion_eligible", "validation",
    }
    if set(plan) != exact_keys:
        raise BundleError("trusted runner dry-run exact schema differs")
    facts = {
        "schema_version": "ullm.aq4_p2_resident_batch.v1", "status": "dry_run", "scope": "full_model",
        "case_count": 1, "warmup_runs": 2, "measured_runs": 10, "transaction_count": 12,
        "prompt_tokens_across_transactions": 1536, "resident_model_loads": 1,
        "execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False,
    }
    if any(plan.get(key) != value for key, value in facts.items()):
        raise BundleError("trusted runner one-case smoke facts differ")
    identity_raw = expected.payloads["identity.json"]
    identity = parse_json(identity_raw, "identity")
    policy_raw = expected.payloads["policy.json"]
    baseline = {
        "run_id": "p2-r9700-resident-one-case-smoke-runner-validate-v4",
        "kind": "active-production",
        "identity_file": {"path": str(CANONICAL_ROOT / "identity.json"), "sha256": sha_bytes(identity_raw)},
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "worker_binary_sha256": EXPECTED_WORKER_SHA,
        "build_git_commit": DRIVER_COMMIT,
    }
    links = {
        "expanded": {"path": str(CANONICAL_ROOT / "case-binding.json"), "sha256": sha_bytes(expected.payloads["case-binding.json"])},
        "fixture_index": {"path": str(CANONICAL_ROOT / "fixture-index.json"), "sha256": sha_bytes(expected.payloads["fixture-index.json"])},
        "policy": {"path": str(CANONICAL_ROOT / "policy.json"), "sha256": sha_bytes(policy_raw)},
    }
    if plan.get("baseline_identity") != baseline or plan.get("links") != links:
        raise BundleError("trusted runner dry-run identity/links differ")
    validation = plan.get("validation")
    if not isinstance(validation, dict) or set(validation) != {"mode", "bundle", "fake_ready", "driver_fake_handshake", "resident_session_id", "driver_identity"}:
        raise BundleError("trusted runner validation exact schema differs")
    expected_validation = {
        "mode": "validate_only",
        "bundle": {"path": str(CANONICAL_ROOT / "bundle.json"), "sha256": sha_bytes(bundle_raw)},
        "fake_ready": {"path": str(CANONICAL_ROOT / "fake-ready.json"), "sha256": sha_bytes(expected.payloads["fake-ready.json"])},
        "driver_fake_handshake": "passed",
        "resident_session_id": "offline-fake-ready-not-executed",
        "driver_identity": identity["resident_driver_identity"],
    }
    if validation != expected_validation:
        raise BundleError("trusted runner validate-only/fake-ready handshake differs")
    return plan


def make_runner_evidence(plan_raw: bytes, stdout: bytes, stderr: bytes, exit_code: int) -> bytes:
    value = {
        "schema_version": "ullm.aq4_p2_resident_runner_subprocess_evidence.v1",
        "runner_source_commit": RUNNER_COMMIT,
        "runner_source_sha256": RUNNER_SOURCE_SHA,
        "runner_subprocess_count": 1,
        "command": runner_validate_argv(),
        "exit_code": exit_code,
        "stdout": {"sha256": sha_bytes(stdout), "utf8": stdout.decode("utf-8")},
        "stderr": {"sha256": sha_bytes(stderr), "utf8": stderr.decode("utf-8")},
        "plan": {"path": str(CANONICAL_ROOT / "dry-run.json"), "sha256": sha_bytes(plan_raw)},
        "facts": {"case_count": 1, "transaction_count": 12, "warmup_runs": 2, "measured_runs": 10, "smoke_only": True, "promotion_eligible": False, "validation_mode": "validate_only", "fake_handshake": "passed"},
        "normal_profile": {"case_count": 84, "separate": True},
    }
    return pretty(value)


def finalize(expected: Reconstruction, plan_raw: bytes, evidence_raw: bytes) -> Reconstruction:
    bundle_raw = pretty(expected.bundle)
    validate_runner_plan(plan_raw, bundle_raw, expected)
    evidence = parse_json(evidence_raw, "trusted runner subprocess evidence")
    expected_evidence = make_runner_evidence(plan_raw, b"", b"", 0)
    if evidence_raw != expected_evidence or evidence != parse_json(expected_evidence, "expected runner evidence"):
        raise BundleError("trusted runner subprocess evidence differs")
    payloads = dict(expected.payloads)
    payloads["dry-run.json"] = plan_raw
    payloads["runner-dry-run-evidence.json"] = evidence_raw
    lines = []
    for name in sorted([*REQUIRED_FILES, "bundle.json"]):
        raw = bundle_raw if name == "bundle.json" else (None if name == "resident-driver" else payloads[name])
        digest = EXPECTED_DRIVER_SHA if raw is None else sha_bytes(raw)
        lines.append(f"{digest}  {name}\n")
    return Reconstruction(payloads, expected.bundle, "".join(lines).encode("ascii"), expected.snapshot)


def _read_bundle_member(
    root: Path,
    name: str,
    label: str,
    pinned_map: PinnedFdMap | None,
    *,
    maximum: int | None = MAX_JSON,
    collect: bool = True,
) -> tuple[bytes, str]:
    path = safe_member(root, name)
    if pinned_map is None:
        if collect:
            raw = read_stable(path, label, maximum, single_link=True)
            return raw, sha_bytes(raw)
        return b"", sha_file(path, label)
    item = pinned_map.binding(path)
    if item is None:
        raise BundleError(f"bundle member is absent from pinned FD map: {name}")
    raw, digest, _metadata = pinned_map.read(item, maximum=maximum, collect=collect)
    return raw, digest


def validate(
    root: Path,
    trusted: Reconstruction | None = None,
    pinned_map: PinnedFdMap | None = None,
) -> dict[str, Any]:
    root = root.absolute()
    root_descriptor: int | None = None
    root_binding: dict[str, Any] | None = None
    if pinned_map is None:
        reject_symlink_components(root, "bundle root")
        if root.is_symlink() or not root.is_dir():
            raise BundleError("bundle root must be a non-symlink directory")
        root_metadata = root.lstat()
        names_before = {entry.name for entry in root.iterdir()}
    else:
        if not root.is_absolute() or ".." in root.parts:
            raise BundleError("logical bundle root must be absolute without parent traversal")
        root_binding = pinned_map.role("bundle_root")
        if (
            root_binding["logical_path"] != str(root)
            or root_binding["kind"] != "directory"
            or root_binding["closure"] != "data_integrity"
            or root_binding["method"] != "pre_post_guard"
        ):
            raise BundleError("pinned bundle root binding differs")
        root_descriptor = root_binding["descriptor"]
        root_metadata = pinned_map.verify_binding(root_binding)
        try:
            logical_root_metadata = os.lstat(root)
        except OSError as error:
            raise BundleError(f"logical bundle root is unavailable: {error}") from error
        if (
            named_identity(logical_root_metadata) != root_binding["identity"]
            or named_identity(root_metadata) != root_binding["identity"]
            or not stat.S_ISDIR(root_metadata.st_mode)
        ):
            raise BundleError("logical and pinned bundle root identity differs")
        names_before = set(os.listdir(root_descriptor))
    root_before = fingerprint(root_metadata)
    if stat.S_IMODE(root_metadata.st_mode) != BUNDLE_ROOT_MODE:
        raise BundleError("bundle root mode differs")
    expected = reconstruct() if trusted is None else trusted
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    if names_before != allowed:
        raise BundleError("bundle directory exact coverage differs")
    initial: dict[str, tuple[int, ...]] = {}
    for name in sorted(allowed):
        observed = (
            safe_member(root, name).lstat()
            if root_descriptor is None
            else os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        )
        mode = REQUIRED_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != mode:
            raise BundleError(f"bundle member type/link/mode differs: {name}")
        initial[name] = fingerprint(observed)
    for name, raw in expected.payloads.items():
        actual, _digest = _read_bundle_member(root, name, name, pinned_map)
        if name.endswith(".json"):
            parse_json(actual, name)
        if actual != raw:
            raise BundleError(f"independent semantic reconstruction differs: {name}")
    _raw, driver_sha = _read_bundle_member(
        root, "resident-driver", "resident driver", pinned_map, maximum=None, collect=False
    )
    if driver_sha != EXPECTED_DRIVER_SHA:
        raise BundleError("detached resident driver expected SHA differs")
    bundle_raw, _digest = _read_bundle_member(root, "bundle.json", "bundle", pinned_map)
    bundle = parse_json(bundle_raw, "bundle")
    if bundle != expected.bundle or bundle_raw != pretty(expected.bundle):
        raise BundleError("independent semantic reconstruction differs: bundle.json")
    plan_raw, _digest = _read_bundle_member(root, "dry-run.json", "dry-run.json", pinned_map)
    evidence_raw, _digest = _read_bundle_member(
        root, "runner-dry-run-evidence.json", "runner-dry-run-evidence.json", pinned_map
    )
    expected = finalize(expected, plan_raw, evidence_raw)
    validate_launch_command(parse_json(expected.payloads["launch-command.json"], "launch command"))
    sums, _digest = _read_bundle_member(root, "SHA256SUMS", "SHA256SUMS", pinned_map)
    if sums != expected.sums:
        raise BundleError("SHA256SUMS independent exact coverage differs")
    for line in sums.decode("ascii").splitlines():
        digest, name = line.split("  ", 1)
        safe_member(root, name)
        _raw, observed_digest = _read_bundle_member(
            root, name, name, pinned_map, maximum=None, collect=False
        )
        if SHA_RE.fullmatch(digest) is None or observed_digest != digest:
            raise BundleError(f"SHA256SUMS differs: {name}")
    if _VALIDATION_HOOK is not None:
        _VALIDATION_HOOK(root)
    if root_descriptor is None:
        final_names = {entry.name for entry in root.iterdir()}
        final_root = root.lstat()
    else:
        final_names = set(os.listdir(root_descriptor))
        assert root_binding is not None
        final_root = pinned_map.verify_binding(root_binding)
        try:
            final_logical_root = os.lstat(root)
        except OSError as error:
            raise BundleError(f"logical bundle root disappeared: {error}") from error
        if (
            named_identity(final_logical_root) != root_binding["identity"]
            or named_identity(final_root) != root_binding["identity"]
        ):
            raise BundleError("late logical bundle root mutation detected")
    if final_names != allowed or fingerprint(final_root) != root_before:
        raise BundleError("late bundle directory mutation detected")
    for name, before in initial.items():
        try:
            after = (
                safe_member(root, name).lstat()
                if root_descriptor is None
                else os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            )
        except FileNotFoundError as error:
            raise BundleError(f"TOCTOU mutation detected: {name}") from error
        if fingerprint(after) != before:
            raise BundleError(f"TOCTOU mutation detected: {name}")
    if pinned_map is not None:
        pinned_map.verify_data_guards()
    expected.snapshot.verify()
    return bundle


def prepare(output: Path, driver_path: Path) -> dict[str, Any]:
    if output.resolve() != CANONICAL_ROOT.resolve():
        raise BundleError(f"output must be the canonical run root: {CANONICAL_ROOT}")
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    expected = reconstruct()
    driver_path = driver_path.resolve()
    if (
        sha_file(driver_path, "clean release resident driver", single_link=False)
        != EXPECTED_DRIVER_SHA
        or driver_path.stat().st_size != EXPECTED_DRIVER_BYTES
    ):
        raise BundleError("release driver differs from trusted clean-build SHA")
    output.mkdir(parents=True)
    for name, raw in expected.payloads.items():
        (output / name).write_bytes(raw)
    with driver_path.open("rb") as source, (output / "resident-driver").open("xb") as target:
        shutil.copyfileobj(source, target, CHUNK)
        target.flush()
        os.fsync(target.fileno())
    (output / "bundle.json").write_bytes(pretty(expected.bundle))
    if RUNNER_VALIDATE_OUTPUT.exists() or RUNNER_VALIDATE_OUTPUT.is_symlink():
        raise BundleError(f"trusted runner validate-only output already exists: {RUNNER_VALIDATE_OUTPUT}")
    completed = None
    try:
        completed = subprocess.run(runner_validate_argv(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        plan_path = RUNNER_VALIDATE_OUTPUT / "resident-batch.plan.json"
        if completed.returncode != 0:
            raise BundleError(f"trusted runner validate-only subprocess failed with exit {completed.returncode}")
        plan_raw = read_stable(plan_path, "trusted runner generated plan", single_link=True)
    finally:
        if RUNNER_VALIDATE_OUTPUT.exists() and not RUNNER_VALIDATE_OUTPUT.is_symlink():
            shutil.rmtree(RUNNER_VALIDATE_OUTPUT)
    if completed is None:
        raise BundleError("trusted runner validate-only subprocess did not run")
    (output / "dry-run.json").write_bytes(plan_raw)
    evidence_raw = make_runner_evidence(plan_raw, completed.stdout, completed.stderr, completed.returncode)
    (output / "runner-dry-run-evidence.json").write_bytes(evidence_raw)
    expected = finalize(expected, plan_raw, evidence_raw)
    (output / "SHA256SUMS").write_bytes(expected.sums)
    for name, (mode, _) in REQUIRED_FILES.items():
        os.chmod(output / name, mode)
    os.chmod(output / "bundle.json", 0o444)
    os.chmod(output / "SHA256SUMS", 0o444)
    os.chmod(output, BUNDLE_ROOT_MODE)
    return validate(output, expected)


def binding_sources(validator_commit: str, validator_sha: str) -> tuple[bytes, bytes, str, str]:
    tree = subprocess.run(["git", "rev-parse", f"{BINDING_SOURCE_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if tree.returncode != 0 or tree.stdout.strip() != BINDING_SOURCE_TREE:
        raise BundleError("binding source commit/tree differs")
    expected_objects = {
        "tools/run-aq4-p2-resident-batch.py": BINDING_RUNNER_GIT_BLOB,
        "crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs": BINDING_DRIVER_GIT_BLOB,
    }
    for path, object_id in expected_objects.items():
        observed = subprocess.run(["git", "rev-parse", f"{BINDING_SOURCE_COMMIT}:{path}"], cwd=ROOT, text=True, capture_output=True, check=False)
        if observed.returncode != 0 or observed.stdout.strip() != object_id:
            raise BundleError(f"binding Git blob object differs: {path}")
    runner = git_blob("tools/run-aq4-p2-resident-batch.py", BINDING_RUNNER_SHA, BINDING_SOURCE_COMMIT)
    if re.fullmatch(r"[0-9a-f]{40}", validator_commit) is None or SHA_RE.fullmatch(validator_sha) is None:
        raise BundleError("binding validator commit/SHA is invalid")
    validator_tree_process = subprocess.run(["git", "rev-parse", f"{validator_commit}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    validator_object_process = subprocess.run(["git", "rev-parse", f"{validator_commit}:tools/prepare-aq4-p2-resident-smoke-bundle.py"], cwd=ROOT, text=True, capture_output=True, check=False)
    if validator_tree_process.returncode != 0 or validator_object_process.returncode != 0:
        raise BundleError("binding validator Git commit/blob is unavailable")
    validator_tree = validator_tree_process.stdout.strip()
    validator_object = validator_object_process.stdout.strip()
    validator = git_blob("tools/prepare-aq4-p2-resident-smoke-bundle.py", validator_sha, validator_commit)
    current_driver = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, BINDING_SOURCE_COMMIT)
    normative_driver = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, DRIVER_COMMIT)
    if current_driver != normative_driver:
        raise BundleError("binding commit changed the normative resident driver blob")
    for contract in (b"ONE_CASE_ROOT_CONTRACT", b"def _run_bundle_validator", b'_require_absolute_nonsymlink_path(path, "trusted bundle validator")', b"nargs=argparse.REMAINDER", b"resident_driver_argv", b"fake_driver_subprocess_count", b"--bundle-root", b"--live-preflight", b"def validate_prepared_preflight_link", b"def require_prepared_preflight_link", b"def validate_live_preflight", b"def require_live_preflight_link", b"def verify_live_preflight", b"prepared_preflight_link = validate_prepared_preflight_link", b"live_preflight_link: LivePreflightLink | None", b"def validate_historical_synthetic_ready_fixture", b"HISTORICAL_SYNTHETIC_READY_FIELDS", b"LIVE_READY_FIELDS", b"HISTORICAL_SYNTHETIC_READY_SCOPE", b'"stage": "pre_spawn_fixture_only"', b"_ready_predicates(value, LIVE_READY_FIELDS)", b"served_binding = _validate_served_model_binding("):
        if contract not in runner:
            raise BundleError("binding runner generic root/validator contract differs")
    for forbidden in (b"preflight_link = live_preflight_link", b"allow_pre_binding_fixture"):
        if forbidden in runner:
            raise BundleError("binding runner prepared/live or synthetic/live separation differs")
    return runner, validator, validator_tree, validator_object


def binding_runner_argv(validator_sha: str, root: Path = BINDING_ROOT) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), str(root / "trusted-runner.py"),
        "--expanded", str(CANONICAL_ROOT / "case-binding.json"),
        "--fixture-index", str(CANONICAL_ROOT / "fixture-index.json"),
        "--identity", str(CANONICAL_ROOT / "identity.json"),
        "--preflight", str(CANONICAL_ROOT / "preflight.json"),
        "--policy", str(CANONICAL_ROOT / "policy.json"),
        "--bundle-root", str(CANONICAL_ROOT),
        "--trusted-validator", str(BINDING_VALIDATOR_EXEC),
        "--trusted-validator-sha256", validator_sha,
        "--output-dir", str(BINDING_RUNNER_OUTPUT),
        "--run-id", "p2-r9700-resident-one-case-smoke-binding-v7-validate",
        "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run",
    ]


def input_root_inventory() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root_metadata = CANONICAL_ROOT.lstat()
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    if {entry.name for entry in CANONICAL_ROOT.iterdir()} != allowed:
        raise BundleError("binding input root exact member coverage differs")
    members: dict[str, dict[str, Any]] = {}
    for name in sorted(allowed):
        path = CANONICAL_ROOT / name
        metadata = path.lstat()
        if name in REQUIRED_FILES:
            mode, role = REQUIRED_FILES[name]
        else:
            mode, role = 0o444, {"bundle.json": "bundle_manifest", "SHA256SUMS": "sha256_manifest"}[name]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != mode:
            raise BundleError(f"binding input root member differs: {name}")
        members[name] = {"path": str(path), "sha256": sha_file(path, f"binding input {name}"), "role": role, "type": "regular_file", "nlink": 1, "mode": f"{mode:04o}"}
    directory = {"path": str(CANONICAL_ROOT), "device": root_metadata.st_dev, "inode": root_metadata.st_ino}
    return directory, members


def validator_report_stdout(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=True, sort_keys=True) + "\n").encode()


def validate_binding_plan(raw: bytes, report: dict[str, Any], directory: dict[str, Any], members: dict[str, dict[str, Any]], validator_sha: str) -> dict[str, Any]:
    plan = parse_json(raw, "binding runner plan")
    exact_keys = {
        "schema_version", "status", "scope", "case_count", "warmup_runs", "measured_runs", "transaction_count",
        "prompt_tokens_across_transactions", "resident_model_loads", "baseline_identity", "links", "execution_mode",
        "smoke_only", "promotion_eligible", "validation",
    }
    if set(plan) != exact_keys:
        raise BundleError("binding runner plan exact schema differs")
    facts = {"schema_version": "ullm.aq4_p2_resident_batch.v1", "status": "dry_run", "scope": "full_model", "case_count": 1, "warmup_runs": 2, "measured_runs": 10, "transaction_count": 12, "prompt_tokens_across_transactions": 1536, "resident_model_loads": 1, "execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False}
    if any(plan.get(key) != value for key, value in facts.items()):
        raise BundleError("binding runner plan one-case facts differ")
    identity = parse_json(read_stable(CANONICAL_ROOT / "identity.json", "binding identity"), "binding identity")
    baseline = {
        "run_id": "p2-r9700-resident-one-case-smoke-binding-v7-validate", "kind": "active-production",
        "identity_file": {"path": str(CANONICAL_ROOT / "identity.json"), "sha256": members["identity.json"]["sha256"]},
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "build_git_commit": DRIVER_COMMIT,
    }
    links = {
        "expanded": {"path": str(CANONICAL_ROOT / "case-binding.json"), "sha256": members["case-binding.json"]["sha256"]},
        "fixture_index": {"path": str(CANONICAL_ROOT / "fixture-index.json"), "sha256": members["fixture-index.json"]["sha256"]},
        "policy": {"path": str(CANONICAL_ROOT / "policy.json"), "sha256": members["policy.json"]["sha256"]},
    }
    if plan.get("baseline_identity") != baseline or plan.get("links") != links:
        raise BundleError("binding runner plan identity/links differ")
    validation = plan.get("validation")
    expected_validator = {
        "subprocess_count": 1,
        "source": {"path": str(BINDING_VALIDATOR_EXEC), "sha256": validator_sha},
        "stdout_sha256": sha_bytes(validator_report_stdout(report)),
        "report_sha256": sha_bytes(canonical(report)),
        "report": report,
    }
    expected_validation = {
        "mode": "validate_only", "root_contract": "ullm.aq4_p2_resident_smoke_bundle_root.v4",
        "bundle_root": directory, "members": members,
        "bundle": {"path": str(CANONICAL_ROOT / "bundle.json"), "sha256": members["bundle.json"]["sha256"]},
        "fake_ready": {"path": str(CANONICAL_ROOT / "fake-ready.json"), "sha256": members["fake-ready.json"]["sha256"]},
        "fake_driver_subprocess_count": 1, "driver_fake_handshake": "passed",
        "fake_ready_scope": BINDING_FAKE_READY_SCOPE,
        "resident_session_id": "offline-fake-ready-not-executed", "driver_identity": identity["resident_driver_identity"],
        "resident_driver_argv": resident_driver_argv(),
        "trusted_bundle_validator": expected_validator,
    }
    if validation != expected_validation:
        raise BundleError("binding runner root/fake-ready/validator report differs")
    return plan


def binding_evidence(plan_raw: bytes, report: dict[str, Any], stdout: bytes, stderr: bytes, exit_code: int, validator_sha: str) -> bytes:
    return pretty({
        "schema_version": "ullm.aq4_p2_resident_binding_runner_evidence.v2", "runner_subprocess_count": 1,
        "command": binding_runner_argv(validator_sha), "exit_code": exit_code,
        "runner_source": {
            "source_commit": BINDING_SOURCE_COMMIT,
            "source_tree": BINDING_SOURCE_TREE,
            "git_blob": BINDING_RUNNER_GIT_BLOB,
            "source_sha256": BINDING_RUNNER_SHA,
            "archive_path": str(BINDING_ROOT / "trusted-runner.py"),
        },
        "stdout": {"sha256": sha_bytes(stdout), "utf8": stdout.decode("utf-8")},
        "stderr": {"sha256": sha_bytes(stderr), "utf8": stderr.decode("utf-8")},
        "plan": {"path": str(BINDING_ROOT / "runner-plan.json"), "sha256": sha_bytes(plan_raw)},
        "trusted_validator": {"source_sha256": validator_sha, "subprocess_count": 1, "canonical_report_sha256": sha_bytes(canonical(report)), "report_file_sha256": sha_bytes(pretty(report))},
    })


def binding_manifest(plan_raw: bytes, evidence_raw: bytes, report_raw: bytes, directory: dict[str, Any], members: dict[str, dict[str, Any]], runner_raw: bytes, validator_raw: bytes, validator_commit: str, validator_tree: str, validator_object: str) -> dict[str, Any]:
    root_fingerprint = {"directory": directory, "members": members, "sha256": sha_bytes(canonical({"directory": directory, "members": members}))}
    return {
        "schema_version": "ullm.aq4_p2_resident_smoke_binding.v7", "status": "prepared_not_executed", "promotion": False,
        "launch_eligible": False, "requires_immutable_launcher": True,
        "binding_root_contract": {
            "type": "directory",
            "mode": f"{BINDING_ROOT_MODE:04o}",
            "members_single_link": True,
            "members_read_only": True,
        },
        "predecessor": {"commit": BINDING_PREDECESSOR_COMMIT, "status": "SUPERSEDED", "execution_eligible": False},
        "trust_roots": {
            "source_commit": BINDING_SOURCE_COMMIT, "source_tree": BINDING_SOURCE_TREE,
            "runner": {
                "source_commit": BINDING_SOURCE_COMMIT,
                "source_tree": BINDING_SOURCE_TREE,
                "git_blob": BINDING_RUNNER_GIT_BLOB,
                "source_sha256": sha_bytes(runner_raw),
                "archive_path": str(BINDING_ROOT / "trusted-runner.py"),
            },
            "validator": {"source_commit": validator_commit, "source_tree": validator_tree, "git_blob": validator_object, "sha256": sha_bytes(validator_raw), "archive_path": str(BINDING_ROOT / "trusted-validator.py"), "execution_path": str(BINDING_VALIDATOR_EXEC)},
            "resident_driver": {"normative_commit": DRIVER_COMMIT, "source_tree": DRIVER_TREE, "git_blob_at_binding_commit": BINDING_DRIVER_GIT_BLOB, "source_sha256": DRIVER_SOURCE_SHA, "build_inputs": DRIVER_BUILD_INPUTS, "blob_unchanged": True, "binary_sha256": EXPECTED_DRIVER_SHA, "binary_bytes": EXPECTED_DRIVER_BYTES, "binary_build_id_sha1": EXPECTED_DRIVER_BUILD_ID, "build": DRIVER_BUILD_METADATA},
        },
        "runner_roles": {
            "prepared_bootstrap": {
                "commit": RUNNER_COMMIT,
                "sha256": RUNNER_SOURCE_SHA,
                "role": "historical_control_member",
                "execution_closure": "control_input/read",
            },
            "binding_actual": {
                "commit": BINDING_SOURCE_COMMIT,
                "sha256": BINDING_RUNNER_SHA,
                "role": "actual_generic_runner",
                "execution_closure": "code_execution/exec",
            },
            "same_runner": False,
        },
        "input_root": root_fingerprint,
        "outputs": {"runner_plan_sha256": sha_bytes(plan_raw), "runner_evidence_sha256": sha_bytes(evidence_raw), "validator_report_file_sha256": sha_bytes(report_raw), "validator_report_canonical_sha256": sha_bytes(canonical(parse_json(report_raw, "binding validator report")))},
        "execution": {"runner_subprocess_count": 1, "trusted_validator_subprocess_count": 1, "fake_driver_subprocess_count": 1, "model_load_executed": False, "gpu_command_executed": False, "service_touched": False},
        "cycle_control": {"input_root_unchanged_after_runner": True, "generated_outputs_outside_input_root": True, "input_root_dry_run_not_replaced": True, "generic_runner_schema_not_embedded_back_into_input_root": True},
        "next_stage": {"name": "L immutable launcher", "required": True, "must_pin": ["input root fingerprint", "binding manifest SHA-256", "runner SHA-256", "validator SHA-256"]},
    }


def validate_binding(validator_commit: str, validator_sha: str, root: Path = BINDING_ROOT) -> dict[str, Any]:
    root = root.absolute()
    reject_symlink_components(root, "binding root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("binding root must be a non-symlink directory")
    root_metadata = root.lstat()
    if stat.S_IMODE(root_metadata.st_mode) != BINDING_ROOT_MODE:
        raise BundleError("binding root mode differs")
    root_before = fingerprint(root_metadata)
    allowed = set(BINDING_FILES) | {"SHA256SUMS"}
    if {entry.name for entry in root.iterdir()} != allowed:
        raise BundleError("binding sidecar exact member coverage differs")
    initial: dict[str, tuple[int, ...]] = {}
    for name in sorted(allowed):
        metadata = (root / name).lstat()
        mode = BINDING_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != mode:
            raise BundleError(f"binding sidecar member differs: {name}")
        initial[name] = fingerprint(metadata)
    runner_raw, validator_raw, validator_tree, validator_object = binding_sources(validator_commit, validator_sha)
    if read_stable(root / "trusted-runner.py", "binding runner") != runner_raw or read_stable(root / "trusted-validator.py", "binding validator") != validator_raw:
        raise BundleError("binding trusted source differs")
    if read_stable(BINDING_VALIDATOR_EXEC, "binding validator execution copy") != validator_raw:
        raise BundleError("binding validator execution copy differs")
    validate(CANONICAL_ROOT)
    directory, members = input_root_inventory()
    plan_raw = read_stable(root / "runner-plan.json", "binding runner plan")
    report_raw = read_stable(root / "validator-report.json", "binding validator report")
    report = parse_json(report_raw, "binding validator report")
    if report != {"status": "prepared_not_executed", "promotion": False, "run_id": "p2-r9700-resident-one-case-smoke-prepared-v4"} or report_raw != pretty(report):
        raise BundleError("binding validator report differs")
    validate_binding_plan(plan_raw, report, directory, members, validator_sha)
    evidence_raw = read_stable(root / "runner-subprocess-evidence.json", "binding runner evidence")
    if evidence_raw != binding_evidence(plan_raw, report, b"", b"", 0, validator_sha):
        raise BundleError("binding runner subprocess evidence differs")
    manifest = binding_manifest(plan_raw, evidence_raw, report_raw, directory, members, runner_raw, validator_raw, validator_commit, validator_tree, validator_object)
    manifest_raw = read_stable(root / "binding-manifest.json", "binding manifest")
    if manifest_raw != pretty(manifest) or parse_json(manifest_raw, "binding manifest") != manifest:
        raise BundleError("binding manifest differs")
    expected_sums = "".join(f"{sha_file(root / name, f'binding sum {name}')}  {name}\n" for name in sorted(BINDING_FILES)).encode("ascii")
    if read_stable(root / "SHA256SUMS", "binding SHA256SUMS") != expected_sums:
        raise BundleError("binding SHA256SUMS differs")
    if _BINDING_VALIDATION_HOOK is not None:
        _BINDING_VALIDATION_HOOK(root)
    if {entry.name for entry in root.iterdir()} != allowed or fingerprint(root.lstat()) != root_before:
        raise BundleError("late binding sidecar directory mutation detected")
    for name, before in initial.items():
        path = root / name
        try:
            after = path.lstat()
        except FileNotFoundError as error:
            raise BundleError(f"late binding sidecar member mutation detected: {name}") from error
        mode = BINDING_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 or stat.S_IMODE(after.st_mode) != mode or fingerprint(after) != before:
            raise BundleError(f"late binding sidecar member mutation detected: {name}")
    return manifest


def prepare_binding(validator_commit: str, validator_sha: str, output: Path = BINDING_ROOT) -> dict[str, Any]:
    if output.resolve() != BINDING_ROOT.resolve():
        raise BundleError(f"binding output must be canonical: {BINDING_ROOT}")
    if output.exists() or output.is_symlink():
        raise BundleError(f"binding output already exists: {output}")
    validate(CANONICAL_ROOT)
    directory_before, members_before = input_root_inventory()
    runner_raw, validator_raw, validator_tree, validator_object = binding_sources(validator_commit, validator_sha)
    output.mkdir(parents=True)
    (output / "trusted-runner.py").write_bytes(runner_raw)
    (output / "trusted-validator.py").write_bytes(validator_raw)
    if read_stable(BINDING_VALIDATOR_EXEC, "binding validator execution copy") != validator_raw:
        raise BundleError(f"binding validator execution path differs: {BINDING_VALIDATOR_EXEC}")
    os.chmod(output / "trusted-runner.py", 0o444)
    os.chmod(output / "trusted-validator.py", 0o444)
    if BINDING_RUNNER_OUTPUT.exists() or BINDING_RUNNER_OUTPUT.is_symlink():
        raise BundleError(f"binding runner output already exists: {BINDING_RUNNER_OUTPUT}")
    completed = None
    try:
        completed = subprocess.run(binding_runner_argv(validator_sha, output), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            raise BundleError(f"binding runner subprocess failed with exit {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}")
        plan_raw = read_stable(BINDING_RUNNER_OUTPUT / "resident-batch.plan.json", "actual binding runner plan")
    finally:
        if BINDING_RUNNER_OUTPUT.exists() and not BINDING_RUNNER_OUTPUT.is_symlink():
            shutil.rmtree(BINDING_RUNNER_OUTPUT)
    if completed is None:
        raise BundleError("binding runner subprocess did not run")
    plan = parse_json(plan_raw, "actual binding runner plan")
    report = plan.get("validation", {}).get("trusted_bundle_validator", {}).get("report")
    if not isinstance(report, dict):
        raise BundleError("actual binding runner omitted validator report")
    validate_binding_plan(plan_raw, report, directory_before, members_before, validator_sha)
    report_raw = pretty(report)
    evidence_raw = binding_evidence(plan_raw, report, completed.stdout, completed.stderr, completed.returncode, validator_sha)
    manifest = binding_manifest(plan_raw, evidence_raw, report_raw, directory_before, members_before, runner_raw, validator_raw, validator_commit, validator_tree, validator_object)
    generated = {"runner-plan.json": plan_raw, "runner-subprocess-evidence.json": evidence_raw, "validator-report.json": report_raw, "binding-manifest.json": pretty(manifest)}
    for name, raw in generated.items():
        (output / name).write_bytes(raw)
    sums = "".join(f"{sha_file(output / name, f'prepared binding {name}')}  {name}\n" for name in sorted(BINDING_FILES)).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    for name, (mode, _) in BINDING_FILES.items():
        os.chmod(output / name, mode)
    os.chmod(output / "SHA256SUMS", 0o444)
    os.chmod(output, BINDING_ROOT_MODE)
    directory_after, members_after = input_root_inventory()
    if directory_after != directory_before or members_after != members_before:
        raise BundleError("binding input root changed during actual runner validation")
    return validate_binding(validator_commit, validator_sha, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=CANONICAL_ROOT)
    prepare_parser.add_argument("--resident-driver", type=Path, default=DRIVER_BUILD_PATH)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--bundle", type=Path, default=CANONICAL_ROOT)
    binding_prepare_parser = sub.add_parser("prepare-binding")
    binding_prepare_parser.add_argument("--output", type=Path, default=BINDING_ROOT)
    binding_prepare_parser.add_argument("--validator-source-commit", required=True)
    binding_prepare_parser.add_argument("--validator-sha256", required=True)
    binding_validate_parser = sub.add_parser("validate-binding")
    binding_validate_parser.add_argument("--binding", type=Path, default=BINDING_ROOT)
    binding_validate_parser.add_argument("--validator-source-commit", required=True)
    binding_validate_parser.add_argument("--validator-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output, args.resident_driver)
        elif args.command == "validate":
            result = validate(args.bundle, pinned_map=PinnedFdMap.from_environment())
        elif args.command == "prepare-binding":
            result = prepare_binding(args.validator_source_commit, args.validator_sha256, args.output)
        else:
            result = validate_binding(args.validator_source_commit, args.validator_sha256, args.binding)
        summary = {"status": result["status"], "promotion": result["promotion"]}
        summary["run_id"] = result.get("run_id", "p2-r9700-resident-one-case-smoke-binding-v7")
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (BundleError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident smoke bundle {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
