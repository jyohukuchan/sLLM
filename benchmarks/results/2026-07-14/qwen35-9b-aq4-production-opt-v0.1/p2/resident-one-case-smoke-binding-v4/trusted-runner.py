#!/usr/bin/env python3
"""Run the representative AQ4 P2 full-model target profile through one resident driver.

The driver protocol is deliberately tiny and hash-only: one child process announces one model
load, then receives case/run commands.  A real GPU driver can implement this protocol later; the
planner and fake-driver tests are CPU-only and never touch a service or device.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, NamedTuple, TypedDict

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_DRIVER_STDOUT_LINE_BYTES = 1024 * 1024
MAX_DRIVER_STDERR_RETAIN_BYTES = 1024 * 1024
MAX_DRIVER_TAIL_BYTES = 64 * 1024
DRIVER_IO_CHUNK_BYTES = 64 * 1024
DRIVER_CLEANUP_GRACE_SECONDS = 5.0
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SCHEMA = "ullm.aq4_p2_resident_batch.v1"
DRIVER_SCHEMA = "ullm.aq4_p2_resident_driver.v2"
PREPARED_PREFLIGHT_FIELDS = {
    "weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes",
    "temporary_bytes", "vram_headroom_bytes", "gpu_process_snapshot",
}
ONE_CASE_BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v3"
ONE_CASE_ROOT_CONTRACT = "ullm.aq4_p2_resident_smoke_bundle_root.v4"
TRUSTED_ONE_CASE_ID = "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target"
TRUSTED_ONE_CASE_SHA256 = "d83a420476bde889c7c8014d7982fd52e0f61ab09b888f66415d0ac9fb443ae7"
TRUSTED_OFFICIAL_CASE_SHA256 = "bf18481edf53a70efe840243f735c12cb949d127db024b38b89b5974ad77eb5a"
TRUSTED_SOURCE_MANIFEST_SHA256 = "1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea"
ONE_CASE_MEMBER_CONTRACT = {
    "SUPERSEDED-0fd7993.json": (0o444, "historical_non_executable_bundle_record"),
    "case-binding.json": (0o444, "runtime_bound_case"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "fixture-index.json": (0o444, "fixture_index"),
    "fixture.json": (0o444, "fixture"),
    "identity.json": (0o444, "resident_identity"),
    "launch-command.json": (0o444, "exact_resident_launch_command"),
    "official-case.json": (0o444, "trusted_official_expansion_case"),
    "package-manifest.json": (0o444, "package_manifest_snapshot"),
    "policy.json": (0o444, "threshold_policy"),
    "preflight.json": (0o444, "synthetic_preflight"),
    "resident-driver": (0o555, "detached_resident_driver"),
    "runner-dry-run-evidence.json": (0o444, "trusted_runner_subprocess_evidence"),
    "served-model.json": (0o444, "served_model_snapshot"),
    "trust-roots.json": (0o444, "independent_trust_roots"),
    "trusted-runner.py": (0o444, "trusted_one_case_smoke_runner"),
}
ONE_CASE_BUNDLE_FILE_MEMBERS = set(ONE_CASE_MEMBER_CONTRACT) - {"dry-run.json", "runner-dry-run-evidence.json"}
ONE_CASE_ROOT_MEMBERS = set(ONE_CASE_MEMBER_CONTRACT) | {"bundle.json", "SHA256SUMS"}
WARMUP_RUNS = 2
MEASURED_RUNS = 10
READY_IDENTITY_KEYS = {
    "binary_sha256",
    "build_git_commit",
    "protocol",
    "worker_binary_sha256",
    "package_manifest_sha256",
    "package_content_sha256",
    "served_model_manifest_sha256",
    "model_id",
    "model_revision",
    "format_id",
    "implementation_id",
    "runtime_device",
    "guard_set_sha256",
}
RUNTIME_DEVICE_KEYS = {
    "runtime_device_index",
    "device_id",
    "backend",
    "name",
    "architecture",
}
DEFAULT_LOCK_PATH = Path("/run/ullm/r9700.lock")
ROCTX_SCHEMA = "ullm.aq4_p2_resident_roctx_ranges.v1"
ROCTX_MARKER_PREFIX = "ullm.aq4_p2.run.v1"
FD_MAP_SCHEMA = "ullm.aq4_p3_inherited_fd_map.v1"
FD_MAP_ENV = "ULLM_AQ4_PINNED_FD_MAP"
FD_MAP_MAX_BYTES = 1024 * 1024
FD_CLOSURE_CONTRACT = {
    "code_execution_closure": "pinned_fd",
    "control_input_closure": "pinned_fd",
    "device_lock_closure": "pinned_fd",
    "data_integrity": "trusted_pre_post_guarded",
}


class BatchError(ValueError):
    pass


class PreparedPreflightLink(TypedDict):
    path: str
    sha256: str


class LivePreflightLink(TypedDict):
    path: str
    sha256: str
    device: int
    inode: int
    captured_unix_ns: int
    runtime_mapping: dict[str, Any]
    lock: dict[str, Any]
    vram: dict[str, Any]


class DriverProtocolError(BatchError):
    def __init__(self, kind: str, stage: str, message: str) -> None:
        self.kind = kind
        self.stage = stage
        super().__init__(message)


class PathComponentIdentity(NamedTuple):
    path: Path
    identity: tuple[int, ...]
    symlink_target: str | None


class RoctxLibraryIdentity(NamedTuple):
    invocation_path: Path
    resolved_path: Path
    sha256: str
    resolved_identity: tuple[int, ...]
    components: tuple[PathComponentIdentity, ...]
    descriptor: int | None = None

    def verify(self) -> None:
        if self.descriptor is not None:
            if ACTIVE_FD_MAP is None:
                raise BatchError("ROCTx pinned FD map disappeared")
            item = ACTIVE_FD_MAP.binding(self.invocation_path, method="dlopen")
            if item is None or item["descriptor"] != self.descriptor:
                raise BatchError("ROCTx pinned FD binding disappeared")
            metadata = ACTIVE_FD_MAP.verify_binding(item)
            _raw, digest, _metadata = read_regular(
                self.invocation_path, "ROCTx pinned library", collect=False
            )
            if digest != self.sha256 or _file_identity(metadata) != self.resolved_identity:
                raise BatchError("ROCTx pinned library identity changed")
            return
        for component in self.components:
            try:
                current = os.lstat(component.path)
            except OSError as error:
                raise BatchError(f"ROCTx invocation component disappeared: {component.path}") from error
            if _file_identity(current) != component.identity:
                raise BatchError(f"ROCTx invocation component changed: {component.path}")
            target = os.readlink(component.path) if stat.S_ISLNK(current.st_mode) else None
            if target != component.symlink_target:
                raise BatchError(f"ROCTx invocation symlink target changed: {component.path}")
        try:
            resolved = self.invocation_path.resolve(strict=True)
        except OSError as error:
            raise BatchError(f"ROCTx invocation path resolution failed: {error}") from error
        if resolved != self.resolved_path:
            raise BatchError("ROCTx invocation path resolved target changed")
        _raw, digest, metadata = read_regular(resolved, "ROCTx resolved library")
        if _file_identity(metadata) != self.resolved_identity or digest != self.sha256:
            raise BatchError("ROCTx resolved library identity changed")


class RoctxRangeRecorder:
    def __init__(self, identity: RoctxLibraryIdentity, handle: Any) -> None:
        self.identity = identity
        try:
            self._push = handle.roctxRangePushA
            self._pop = handle.roctxRangePop
        except AttributeError as error:
            raise BatchError("ROCTx library lacks roctxRangePushA/roctxRangePop") from error
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int
        self.pid = os.getpid()
        self.thread_id = threading.get_ident()
        self.active: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    @classmethod
    def load(
        cls,
        invocation_path: Path,
        expected_sha256: str,
        *,
        cdll_factory: Any = ctypes.CDLL,
    ) -> "RoctxRangeRecorder":
        if not invocation_path.is_absolute() or ".." in invocation_path.parts:
            raise BatchError("ROCTx invocation path must be absolute without parent traversal")
        if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
            raise BatchError("ROCTx expected SHA-256 is invalid")
        mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(invocation_path, method="dlopen")
        if mapped is not None:
            _raw, digest, metadata = read_regular(
                invocation_path, "ROCTx pinned library", collect=False
            )
            if digest != expected_sha256:
                raise BatchError("ROCTx pinned library SHA-256 differs")
            identity = RoctxLibraryIdentity(
                invocation_path,
                Path(mapped["resolved_path"]),
                digest,
                _file_identity(metadata),
                (),
                mapped["descriptor"],
            )
            try:
                handle = cdll_factory(
                    f"/proc/self/fd/{mapped['descriptor']}",
                    mode=getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_LOCAL", 0),
                )
            except OSError as error:
                raise BatchError(f"ROCTx pinned library load failed: {error}") from error
            identity.verify()
            return cls(identity, handle)
        components: list[PathComponentIdentity] = []
        current = Path(invocation_path.anchor)
        for part in invocation_path.parts[1:]:
            current /= part
            try:
                metadata = os.lstat(current)
            except OSError as error:
                raise BatchError(f"ROCTx invocation component is unavailable: {current}") from error
            if stat.S_ISLNK(metadata.st_mode):
                components.append(
                    PathComponentIdentity(
                        current, _file_identity(metadata), os.readlink(current)
                    )
                )
        try:
            resolved = invocation_path.resolve(strict=True)
        except OSError as error:
            raise BatchError(f"ROCTx invocation path resolution failed: {error}") from error
        try:
            metadata = os.lstat(resolved)
        except OSError as error:
            raise BatchError(f"ROCTx resolved library metadata failed: {error}") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BatchError("ROCTx resolved library must be a single-link regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(resolved, flags)
        except OSError as error:
            raise BatchError(f"ROCTx resolved library open failed: {error}") from error
        digest_value = hashlib.sha256()
        try:
            opened = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(metadata):
                raise BatchError("ROCTx resolved library changed while opening")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest_value.update(chunk)
            digest = digest_value.hexdigest()
            after = os.lstat(resolved)
            if (
                _file_identity(os.fstat(descriptor)) != _file_identity(metadata)
                or _file_identity(after) != _file_identity(metadata)
            ):
                raise BatchError("ROCTx resolved library changed while hashing")
            if digest != expected_sha256:
                raise BatchError("ROCTx resolved library SHA-256 differs")
            identity = RoctxLibraryIdentity(
                invocation_path,
                resolved,
                digest,
                _file_identity(metadata),
                tuple(components),
            )
            try:
                handle = cdll_factory(
                    f"/proc/self/fd/{descriptor}",
                    mode=getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_LOCAL", 0),
                )
            except OSError as error:
                raise BatchError(f"ROCTx library load failed: {error}") from error
            if _file_identity(os.fstat(descriptor)) != identity.resolved_identity:
                raise BatchError("ROCTx resolved library changed while loading")
            identity.verify()
        finally:
            os.close(descriptor)
        recorder = cls(identity, handle)
        return recorder

    def _same_context(self) -> None:
        if os.getpid() != self.pid or threading.get_ident() != self.thread_id:
            raise BatchError("ROCTx ranges must execute in one PID/thread")

    def begin(self, name: str, run_index: int, run_kind: str) -> None:
        self._same_context()
        if self.active is not None or run_index != len(self.records):
            raise BatchError("ROCTx range begin order/unbalanced state differs")
        expected_kind = "warmup" if run_index < WARMUP_RUNS else "measured"
        if run_index >= WARMUP_RUNS + MEASURED_RUNS or run_kind != expected_kind:
            raise BatchError("ROCTx range index/kind differs")
        result = self._push(name.encode("utf-8"))
        if type(result) is not int or result < 0:
            raise BatchError("ROCTx range push failed")
        self.active = {
            "name": name,
            "run_index": run_index,
            "run_kind": run_kind,
            "push_result": result,
        }

    def end(self) -> None:
        self._same_context()
        if self.active is None:
            raise BatchError("ROCTx range pop has no active range")
        result = self._pop()
        if type(result) is not int or result < 0:
            raise BatchError("ROCTx range pop failed")
        self.records.append({**self.active, "pop_result": result})
        self.active = None

    @contextmanager
    def range(self, name: str, run_index: int, run_kind: str) -> Iterator[None]:
        self.begin(name, run_index, run_kind)
        try:
            yield
        finally:
            self.end()

    def close_active(self) -> None:
        if self.active is not None:
            self.end()

    def evidence(self) -> dict[str, Any]:
        self._same_context()
        if self.active is not None or len(self.records) != WARMUP_RUNS + MEASURED_RUNS:
            raise BatchError("ROCTx range audit is incomplete or unbalanced")
        for index, record in enumerate(self.records):
            expected_kind = "warmup" if index < WARMUP_RUNS else "measured"
            if record["run_index"] != index or record["run_kind"] != expected_kind:
                raise BatchError("ROCTx range audit order differs")
        self.identity.verify()
        value = {
            "schema_version": ROCTX_SCHEMA,
            "status": "complete_diagnostic",
            "measurement_eligible": False,
            "promotion_eligible": False,
            "audit_sha256": None,
            "pid": self.pid,
            "thread_id": self.thread_id,
            "library": {
                "invocation_path": str(self.identity.invocation_path),
                "resolved_path": str(self.identity.resolved_path),
                "sha256": self.identity.sha256,
                "symbols": ["roctxRangePushA", "roctxRangePop"],
                "components": [
                    {
                        "path": str(item.path),
                        "identity": list(item.identity),
                        "symlink_target": item.symlink_target,
                    }
                    for item in self.identity.components
                ],
            },
            "ranges": self.records,
        }
        value["audit_sha256"] = sha_bytes(canonical(value))
        return value


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise BatchError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _named_file_identity(value: os.stat_result) -> dict[str, int]:
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
    def from_environment(cls, *, required: bool) -> "PinnedFdMap | None":
        raw_descriptor = os.environ.get(FD_MAP_ENV)
        if raw_descriptor is None:
            if required:
                raise BatchError("profile execution requires the pinned FD map")
            return None
        if not raw_descriptor.isascii() or not raw_descriptor.isdecimal():
            raise BatchError("pinned FD map descriptor is invalid")
        descriptor = int(raw_descriptor)
        if descriptor < 3:
            raise BatchError("pinned FD map descriptor is reserved")
        required_seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        try:
            if not required_seals or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals != required_seals:
                raise BatchError("pinned FD map seals differ")
            data = os.pread(descriptor, FD_MAP_MAX_BYTES + 1, 0)
        except OSError as error:
            raise BatchError(f"pinned FD map read failed: {error}") from error
        if len(data) > FD_MAP_MAX_BYTES or not data.endswith(b"\n"):
            raise BatchError("pinned FD map byte contract differs")
        try:
            value = json.loads(
                data[:-1].decode("ascii"),
                object_pairs_hook=pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite FD map number: {item}")),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BatchError(f"pinned FD map JSON is invalid: {error}") from error
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "status", "map_sha256", "logical_argv_sha256",
            "closure_contract", "bindings",
        }:
            raise BatchError("pinned FD map root fields differ")
        if value.get("schema_version") != FD_MAP_SCHEMA or value.get("status") != "bound":
            raise BatchError("pinned FD map schema/status differs")
        declared = value.get("map_sha256")
        clone = json.loads(json.dumps(value))
        clone["map_sha256"] = None
        if not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None or sha_bytes(canonical(clone)) != declared:
            raise BatchError("pinned FD map self-hash differs")
        if value.get("closure_contract") != FD_CLOSURE_CONTRACT:
            raise BatchError("pinned FD map closure contract differs")
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise BatchError("pinned FD map bindings are missing")
        paths: set[str] = set()
        roles: set[str] = set()
        descriptors: set[int] = set()
        allowed = {
            "code_execution": {"exec", "dlopen"},
            "control_input": {"read"},
            "device_lock": {"flock"},
            "data_integrity": {"pre_post_guard"},
        }
        for item in bindings:
            if not isinstance(item, dict) or set(item) != {
                "role", "logical_path", "resolved_path", "descriptor", "kind", "closure", "method",
                "identity", "sha256",
            }:
                raise BatchError("pinned FD map binding fields differ")
            role = item.get("role")
            path = item.get("logical_path")
            child_descriptor = item.get("descriptor")
            identity = item.get("identity")
            closure = item.get("closure")
            method = item.get("method")
            if (
                not isinstance(role, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", role)
                or role in roles
                or not isinstance(path, str)
                or not Path(path).is_absolute()
                or ".." in Path(path).parts
                or path in paths
                or type(child_descriptor) is not int
                or child_descriptor < 3
                or child_descriptor in descriptors
                or closure not in allowed
                or method not in allowed[closure]
                or item.get("kind") not in {"regular_file", "directory", "symlinked_file"}
                or (
                    item.get("kind") == "symlinked_file"
                    and (
                        not isinstance(item.get("resolved_path"), str)
                        or not Path(item["resolved_path"]).is_absolute()
                        or ".." in Path(item["resolved_path"]).parts
                    )
                )
                or (item.get("kind") != "symlinked_file" and item.get("resolved_path") is not None)
                or not isinstance(identity, dict)
                or set(identity) != {"device", "inode", "mode", "nlink", "size", "mtime_ns", "ctime_ns"}
                or any(type(part) is not int for part in identity.values())
                or (item.get("sha256") is not None and (not isinstance(item["sha256"], str) or SHA256_RE.fullmatch(item["sha256"]) is None))
                or (method in {"exec", "dlopen", "read"} and item.get("sha256") is None)
                or (method in {"flock", "pre_post_guard"} and item.get("sha256") is not None)
            ):
                raise BatchError("pinned FD map binding value differs")
            try:
                observed = _named_file_identity(os.fstat(child_descriptor))
            except OSError as error:
                raise BatchError(f"pinned FD binding is unavailable: {role}: {error}") from error
            stable_keys = {"device", "inode", "mode", "nlink"}
            matches = (
                all(observed[key] == identity[key] for key in stable_keys)
                if method == "flock"
                else all(
                    observed[key] == identity[key]
                    for key in {"device", "inode", "mode", "nlink", "size", "mtime_ns"}
                )
                if method != "pre_post_guard"
                else observed == identity
            )
            if not matches:
                raise BatchError(f"pinned FD binding identity differs: {role}")
            paths.add(path)
            roles.add(role)
            descriptors.add(child_descriptor)
        result = cls(descriptor, value)
        result.verify_data_guards()
        return result

    def binding(self, path: Path, *, method: str | None = None) -> dict[str, Any] | None:
        item = self.bindings.get(str(path))
        if item is not None and method is not None and item["method"] != method:
            raise BatchError(f"pinned FD method differs: {item['role']}")
        return item

    def role(self, role: str, *, method: str | None = None) -> dict[str, Any]:
        item = self.roles.get(role)
        if item is None or (method is not None and item["method"] != method):
            raise BatchError(f"required pinned FD role is missing: {role}")
        return item

    def descriptors(self) -> tuple[int, ...]:
        return tuple(sorted({self.descriptor, *(item["descriptor"] for item in self.value["bindings"])}))

    def verify_binding(self, item: dict[str, Any]) -> os.stat_result:
        try:
            metadata = os.fstat(item["descriptor"])
        except OSError as error:
            raise BatchError(f"pinned FD binding disappeared: {item['role']}: {error}") from error
        observed = _named_file_identity(metadata)
        stable_keys = {"device", "inode", "mode", "nlink"}
        matches = (
            all(observed[key] == item["identity"][key] for key in stable_keys)
            if item["method"] == "flock"
            else all(
                observed[key] == item["identity"][key]
                for key in {"device", "inode", "mode", "nlink", "size", "mtime_ns"}
            )
            if item["method"] != "pre_post_guard"
            else observed == item["identity"]
        )
        if not matches:
            raise BatchError(f"pinned FD binding identity changed: {item['role']}")
        return metadata

    def verify_data_guards(self) -> None:
        for item in self.value["bindings"]:
            self.verify_binding(item)
            if item["method"] != "pre_post_guard":
                continue
            path = Path(item["logical_path"])
            try:
                current = os.lstat(path)
            except OSError as error:
                raise BatchError(f"guarded data path is unavailable: {path}: {error}") from error
            if _named_file_identity(current) != item["identity"]:
                raise BatchError(f"guarded data path identity changed: {path}")

    def evidence(self) -> dict[str, Any]:
        return {
            **self.value["closure_contract"],
            "fd_map_schema": FD_MAP_SCHEMA,
            "fd_map_sha256": self.value["map_sha256"],
            "bindings": [
                {
                    key: item[key]
                    for key in ("role", "logical_path", "resolved_path", "kind", "closure", "method", "identity", "sha256")
                }
                for item in self.value["bindings"]
            ],
        }


ACTIVE_FD_MAP: PinnedFdMap | None = None


def effective_fd_path(path: Path, *, method: str, role: str | None = None) -> str:
    if ACTIVE_FD_MAP is None:
        return str(path)
    item = (
        ACTIVE_FD_MAP.role(role, method=method)
        if role is not None
        else ACTIVE_FD_MAP.binding(path, method=method)
    )
    if item is None:
        raise BatchError(f"unmapped {method} path is forbidden: {path}")
    ACTIVE_FD_MAP.verify_binding(item)
    return f"/proc/self/fd/{item['descriptor']}"


def fd_child_options() -> dict[str, Any]:
    if ACTIVE_FD_MAP is None:
        return {}
    return {
        "pass_fds": ACTIVE_FD_MAP.descriptors(),
        "env": dict(os.environ),
    }


def _require_absolute_nonsymlink_path(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise BatchError(f"{label} path must be absolute without parent traversal")
    for parent in reversed(path.parents):
        try:
            if stat.S_ISLNK(os.lstat(parent).st_mode):
                raise BatchError(f"{label} path has a symlink parent")
        except FileNotFoundError:
            continue


def read_regular(path: Path, label: str, maximum: int | None = None, *, absolute: bool = False, collect: bool = True) -> tuple[bytes, str, os.stat_result]:
    mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path)
    if mapped is not None:
        if mapped["kind"] not in {"regular_file", "symlinked_file"} or mapped["method"] not in {"read", "exec", "dlopen"}:
            raise BatchError(f"{label} pinned FD is not readable")
        metadata = ACTIVE_FD_MAP.verify_binding(mapped)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BatchError(f"{label} pinned FD must be a single-link regular file")
        if maximum is not None and metadata.st_size > maximum:
            raise BatchError(f"{label} exceeds the byte bound")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        offset = 0
        while offset < metadata.st_size:
            chunk = os.pread(mapped["descriptor"], min(1024 * 1024, metadata.st_size - offset), offset)
            if not chunk:
                raise BatchError(f"{label} pinned FD ended early")
            offset += len(chunk)
            if maximum is not None and offset > maximum:
                raise BatchError(f"{label} exceeds the byte bound")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        if _file_identity(ACTIVE_FD_MAP.verify_binding(mapped)) != _file_identity(metadata):
            raise BatchError(f"{label} pinned FD changed while reading")
        observed_sha = digest.hexdigest()
        if mapped["sha256"] is not None and observed_sha != mapped["sha256"]:
            raise BatchError(f"{label} pinned FD SHA-256 differs")
        return b"".join(chunks), observed_sha, metadata
    if absolute:
        _require_absolute_nonsymlink_path(path, label)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise BatchError(f"{label} metadata failed: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise BatchError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise BatchError(f"{label} exceeds the byte bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BatchError(f"{label} open failed: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise BatchError(f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise BatchError(f"{label} exceeds the byte bound")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.lstat(path)
        if _file_identity(before) != _file_identity(after) or _file_identity(before) != _file_identity(os.fstat(descriptor)):
            raise BatchError(f"{label} changed while reading")
        return b"".join(chunks), digest.hexdigest(), before
    finally:
        os.close(descriptor)


def load(path: Path, label: str) -> dict[str, Any]:
    data, _, _ = read_regular(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BatchError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str, *, absolute: bool = False) -> str:
    return read_regular(path, label, absolute=absolute, collect=False)[1]


def normalize_fixture_paths(fixture_index: dict[str, Any]) -> None:
    entries = fixture_index.get("cases")
    if not isinstance(entries, list):
        raise BatchError("fixture index cases are missing")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("fixture_path"), str):
            raise BatchError("fixture index contains an invalid fixture path")
        raw = Path(entry["fixture_path"])
        mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(raw, method="read")
        if mapped is None:
            _require_absolute_nonsymlink_path(raw, "fixture")
        elif not raw.is_absolute() or ".." in raw.parts:
            raise BatchError("fixture logical path must be absolute without parent traversal")
        sha_file(raw, "fixture", absolute=True)
        if mapped is not None:
            entry["fixture_path"] = str(raw)
            continue
        try:
            resolved = raw.resolve(strict=True)
        except OSError as error:
            raise BatchError(f"fixture path resolution failed: {error}") from error
        if resolved != raw:
            raise BatchError("fixture path must already be resolved")
        entry["fixture_path"] = str(resolved)


def validate_driver_argv_schema(command: list[str], identity: dict[str, Any], *, expected_argv: list[str] | None = None) -> None:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise BatchError("driver command is empty or invalid")
    bound = identity.get("resident_driver_identity")
    runtime_device = bound.get("runtime_device") if isinstance(bound, dict) else None
    expected_device_index = runtime_device.get("runtime_device_index") if isinstance(runtime_device, dict) else None
    expected_build_commit = bound.get("build_git_commit") if isinstance(bound, dict) else None
    if (
        len(command) != 7
        or command[1] != "--served-model-manifest"
        or not Path(command[2]).is_absolute()
        or ".." in Path(command[2]).parts
        or command[3] != "--device-index"
        or command[4] != str(expected_device_index)
        or command[5] != "--build-git-commit"
        or command[6] != expected_build_commit
    ):
        raise BatchError("resident driver argv does not match the exact production schema")
    if expected_argv is not None and command != expected_argv:
        raise BatchError("resident driver argv differs from the one-case launch binding")


def validate_driver_command(command: list[str], identity: dict[str, Any], *, expected_argv: list[str] | None = None) -> dict[str, Any]:
    validate_driver_argv_schema(command, identity, expected_argv=expected_argv)
    path = Path(command[0])
    _, digest, metadata = read_regular(path, "resident driver executable", absolute=True, collect=False)
    if metadata.st_mode & 0o111 == 0:
        raise BatchError("resident driver executable is not executable")
    bound = identity.get("resident_driver_identity")
    if not isinstance(bound, dict) or bound.get("binary_sha256") != digest:
        raise BatchError("resident driver executable SHA differs from bound identity")
    return {
        "path": str(path),
        "sha256": digest,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "nlink": metadata.st_nlink,
    }


@contextmanager
def acquire_device_lock(
    path: Path,
    run_id: str,
    driver: dict[str, Any],
    *,
    expected_identity: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    pinned = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path, method="flock")
    if pinned is None:
        _require_absolute_nonsymlink_path(path, "device lock")
    elif not path.is_absolute() or ".." in path.parts:
        raise BatchError("device lock logical path must be absolute without parent traversal")
    before: os.stat_result | None = None
    if expected_identity is not None:
        if (
            set(expected_identity) != {"device", "inode"}
            or type(expected_identity.get("device")) is not int
            or expected_identity["device"] < 0
            or type(expected_identity.get("inode")) is not int
            or expected_identity["inode"] < 0
        ):
            raise BatchError("expected device lock identity differs")
        try:
            before = os.lstat(path) if pinned is None else ACTIVE_FD_MAP.verify_binding(pinned)
        except OSError as error:
            raise BatchError(f"bound device lock metadata failed: {error}") from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_gid != os.getegid()
        ):
            raise BatchError("bound device lock file contract differs")
        if before.st_dev != expected_identity["device"] or before.st_ino != expected_identity["inode"]:
            raise BatchError("bound device lock identity differs")
    descriptor_owned = pinned is None
    if pinned is not None:
        descriptor = pinned["descriptor"]
    else:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if expected_identity is None:
            flags |= os.O_CREAT
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise BatchError(f"device lock open failed: {error}") from error
    owner = {
        "schema_version": "ullm.aq4_p2_device_lock_owner.v1",
        "path": str(path),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_id": run_id,
        "acquired_unix_ns": time.time_ns(),
        "driver": driver,
    }
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BatchError("device lock must be a single-link regular file")
        if before is not None and (
            metadata.st_dev != before.st_dev
            or metadata.st_ino != before.st_ino
            or metadata.st_dev != expected_identity["device"]
            or metadata.st_ino != expected_identity["inode"]
        ):
            raise BatchError("bound device lock changed while opening")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchError(f"device lock is already owned: {path}") from error
        if before is not None and pinned is None:
            try:
                current = os.lstat(path)
            except OSError as error:
                raise BatchError(f"bound device lock final metadata failed: {error}") from error
            if current.st_dev != metadata.st_dev or current.st_ino != metadata.st_ino:
                raise BatchError("bound device lock path changed before acquisition")
        owner["device"] = metadata.st_dev
        owner["inode"] = metadata.st_ino
        payload = canonical(owner) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise BatchError("device lock owner metadata write failed")
            offset += written
        os.fsync(descriptor)
        yield owner
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            if descriptor_owned:
                os.close(descriptor)


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return sha_bytes(canonical(value))


def atomic_write(path: Path, value: Any, mode: int | None = None) -> None:
    if os.path.lexists(path):
        raise BatchError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=True, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise BatchError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_bytes(path: Path, raw: bytes, mode: int = 0o444) -> None:
    if os.path.lexists(path):
        raise BatchError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise BatchError(f"atomic byte write failed: {path}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, mode)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise BatchError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def select_target_cases(expanded: dict[str, Any], fixture_index: dict[str, Any], *, one_case_smoke: bool = False) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise BatchError("expanded manifest schema differs")
    if fixture_index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1":
        raise BatchError("fixture index schema differs")
    cases = expanded.get("cases")
    if not isinstance(cases, list):
        raise BatchError("expanded cases are missing")
    selected = [
        case for case in cases
        if isinstance(case, dict)
        and case.get("stage_id") == "representative"
        and case.get("scope") == "full_model"
        and case.get("phase") == "cold_prefill"
        and case.get("device", {}).get("device_id") == "r9700-rdna4"
        and case.get("control_id") == "aq4_0_target"
    ]
    expected_cases = 1 if one_case_smoke else 84
    label = "one-case smoke" if one_case_smoke else "representative full_model target profile"
    if len(selected) != expected_cases:
        raise BatchError(f"{label} must contain exactly {expected_cases} target cases, got {len(selected)}")
    selected_ids = [case.get("case_id") for case in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise BatchError("representative target profile contains duplicate case IDs")
    index_cases = fixture_index.get("cases")
    if not isinstance(index_cases, list):
        raise BatchError("fixture index cases are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in index_cases:
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or not entry["case_id"] or entry["case_id"] in by_id:
            raise BatchError("fixture index contains invalid or duplicate case IDs")
        by_id[entry["case_id"]] = entry
    if len(by_id) != fixture_index.get("case_count"):
        raise BatchError("fixture index case coverage differs")
    for case in selected:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case.get("case_sha256") != case_hash(case):
            raise BatchError(f"selected case identity differs: {case_id}")
        entry = by_id.get(case_id)
        if not isinstance(entry, dict) or entry.get("case_sha256") != case.get("case_sha256") or entry.get("prompt_tokens") != case.get("prompt_tokens") or entry.get("context_tokens") != case.get("context_tokens") or entry.get("generated_tokens") != case.get("generated_tokens"):
            raise BatchError(f"fixture index does not bind selected case: {case_id}")
        fixture_path = Path(entry.get("fixture_path", ""))
        if sha_file(fixture_path, "fixture") != entry.get("fixture_sha256"):
            raise BatchError(f"fixture hash differs: {case_id}")
    return sorted(selected, key=lambda case: case["case_id"])


def _identity_self_sha256(identity: dict[str, Any]) -> str:
    value = json.loads(json.dumps(identity))
    value.pop("_path", None)
    value.pop("_sha256", None)
    value["identity_sha256"] = None
    return sha_bytes(canonical(value))


def _run_fake_ready_handshake(path: Path, timeout: float) -> dict[str, Any]:
    mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path, method="read")
    child = (
        "import os,sys\n"
        "d=sys.stdin.buffer.read(67108865) if sys.argv[1]=='-' else open(sys.argv[1],'rb').read(67108865)\n"
        "if len(d)>67108864: raise SystemExit(91)\n"
        "sys.stdout.buffer.write(d)\n"
        "sys.stdout.buffer.flush()\n"
    )
    pinned_raw = None
    child_path = str(path)
    if mapped is not None:
        pinned_raw, _digest, _metadata = read_regular(path, "one-case smoke fake-ready", MAX_JSON_BYTES)
        child_path = "-"
    completed = subprocess.run(
        [effective_fd_path(Path(sys.executable), method="exec", role="python_interpreter"), "-I", "-c", child, child_path],
        input=pinned_raw,
        stdin=subprocess.DEVNULL if pinned_raw is None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        **fd_child_options(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise BatchError("one-case smoke fake-ready subprocess handshake failed")
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite fake-ready number: {item}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid one-case smoke fake-ready subprocess response: {error}") from error
    if not isinstance(value, dict):
        raise BatchError("one-case smoke fake-ready subprocess response is not an object")
    return value


def _run_bundle_validator(path: Path, expected_sha256: str, root: Path, timeout: float) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise BatchError("trusted bundle validator expected SHA is invalid")
    mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path, method="exec")
    if mapped is None:
        _require_absolute_nonsymlink_path(path, "trusted bundle validator")
    source_raw, source_sha, source_before = read_regular(path, "trusted bundle validator", MAX_JSON_BYTES, absolute=True)
    if source_sha != expected_sha256:
        raise BatchError("trusted bundle validator source differs from expected SHA")
    if not source_raw.startswith(b"#!") and path.suffix != ".py":
        raise BatchError("trusted bundle validator is not a Python source file")
    python_path = effective_fd_path(Path(sys.executable), method="exec", role="python_interpreter")
    validator_path = effective_fd_path(path, method="exec")
    completed = subprocess.run(
        [python_path, validator_path, "validate", "--bundle", str(root)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        **fd_child_options(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise BatchError("trusted bundle validator subprocess rejected the bundle")
    source_after = ACTIVE_FD_MAP.verify_binding(mapped) if mapped is not None else os.lstat(path)
    if _file_identity(source_before) != _file_identity(source_after) or sha_file(path, "trusted bundle validator", absolute=True) != source_sha:
        raise BatchError("trusted bundle validator source changed during subprocess validation")
    try:
        report = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"trusted bundle validator report is invalid: {error}") from error
    if (
        not isinstance(report, dict)
        or set(report) != {"status", "promotion", "run_id"}
        or report.get("status") != "prepared_not_executed"
        or report.get("promotion") is not False
        or not isinstance(report.get("run_id"), str)
        or not report["run_id"]
    ):
        raise BatchError("trusted bundle validator report fields differ")
    return {
        "subprocess_count": 1,
        "source": {"path": str(path), "sha256": source_sha},
        "stdout_sha256": sha_bytes(completed.stdout),
        "report_sha256": sha_bytes(canonical(report)),
        "report": report,
    }


def validate_one_case_smoke_bundle(args: argparse.Namespace, expanded: dict[str, Any], fixture_index: dict[str, Any], identity: dict[str, Any], preflight: dict[str, Any], policy: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.bundle_root is None:
        raise BatchError("--bundle-root is required with --one-case-smoke")
    root_binding = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(
        args.bundle_root, method="pre_post_guard"
    )
    if root_binding is None:
        _require_absolute_nonsymlink_path(args.bundle_root, "one-case smoke bundle root")
        try:
            root = args.bundle_root.resolve(strict=True)
        except OSError as error:
            raise BatchError(f"one-case smoke bundle root resolution failed: {error}") from error
        root_metadata = os.lstat(root)
        root_descriptor = None
    else:
        root = args.bundle_root
        root_metadata = ACTIVE_FD_MAP.verify_binding(root_binding)
        root_descriptor = root_binding["descriptor"]
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise BatchError("one-case smoke bundle root must be a non-symlink directory")
    names = set(os.listdir(root_descriptor)) if root_descriptor is not None else {entry.name for entry in root.iterdir()}
    if names != ONE_CASE_ROOT_MEMBERS:
        raise BatchError("one-case smoke bundle root exact member coverage differs")
    expected_paths = {
        "expanded": root / "case-binding.json",
        "fixture_index": root / "fixture-index.json",
        "identity": root / "identity.json",
        "preflight": root / "preflight.json",
        "policy": root / "policy.json",
    }
    for name, expected in expected_paths.items():
        supplied_path = getattr(args, name)
        supplied = supplied_path if ACTIVE_FD_MAP is not None and ACTIVE_FD_MAP.binding(supplied_path) is not None else supplied_path.resolve(strict=True)
        if supplied != expected:
            raise BatchError(f"one-case smoke {name} is not the bundle root v4 member")
    bundle_path = root / "bundle.json"
    fake_ready_path = root / "fake-ready.json"
    initial: dict[str, tuple[int, ...]] = {}
    member_inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(ONE_CASE_ROOT_MEMBERS):
        member_path = root / name
        member_binding = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(member_path)
        metadata = (
            ACTIVE_FD_MAP.verify_binding(member_binding)
            if member_binding is not None
            else os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            if root_descriptor is not None
            else os.lstat(member_path)
        )
        expected_mode = 0o444 if name in {"bundle.json", "SHA256SUMS"} else ONE_CASE_MEMBER_CONTRACT[name][0]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise BatchError(f"one-case smoke bundle member type/link/mode differs: {name}")
        initial[name] = _file_identity(metadata)
        role = {"bundle.json": "bundle_manifest", "SHA256SUMS": "sha256_manifest"}[name] if name in {"bundle.json", "SHA256SUMS"} else ONE_CASE_MEMBER_CONTRACT[name][1]
        member_inventory[name] = {"path": str(member_path), "sha256": sha_file(member_path, f"one-case smoke {name}"), "role": role, "type": "regular_file", "nlink": metadata.st_nlink, "mode": f"{expected_mode:04o}"}
    bundle = load(bundle_path, "one-case smoke bundle")
    if bundle.get("schema_version") != ONE_CASE_BUNDLE_SCHEMA or bundle.get("status") != "prepared_not_executed" or bundle.get("promotion") is not False:
        raise BatchError("one-case smoke bundle v3 status/promotion differs")
    if bundle.get("canonical_root") != str(root):
        raise BatchError("one-case smoke bundle/root identity differs")
    if len(cases) != 1:
        raise BatchError("one-case smoke internal case count differs")
    case = cases[0]
    if case.get("case_id") != TRUSTED_ONE_CASE_ID or case.get("case_sha256") != TRUSTED_ONE_CASE_SHA256:
        raise BatchError("one-case smoke trusted case ID/hash differs")
    bindings = bundle.get("bindings")
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != ONE_CASE_BUNDLE_FILE_MEMBERS:
        raise BatchError("one-case smoke bundle file role/path coverage differs")
    for name in sorted(ONE_CASE_BUNDLE_FILE_MEMBERS):
        mode, role = ONE_CASE_MEMBER_CONTRACT[name]
        record = files.get(name)
        if not isinstance(record, dict) or set(record) != {"mode", "role", "sha256"} or record.get("mode") != f"{mode:04o}" or record.get("role") != role or not isinstance(record.get("sha256"), str) or SHA256_RE.fullmatch(record["sha256"]) is None:
            raise BatchError(f"one-case smoke bundle file contract differs: {name}")
        if sha_file(root / name, f"one-case smoke {name}") != record["sha256"]:
            raise BatchError(f"one-case smoke bundle file SHA differs: {name}")
    sums_raw, _, _ = read_regular(root / "SHA256SUMS", "one-case smoke SHA256SUMS", MAX_JSON_BYTES)
    try:
        sum_lines = sums_raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise BatchError("one-case smoke SHA256SUMS is not ASCII") from error
    expected_sum_names = set(ONE_CASE_MEMBER_CONTRACT) | {"bundle.json"}
    observed_sums: dict[str, str] = {}
    for line in sum_lines:
        if line.count("  ") != 1:
            raise BatchError("one-case smoke SHA256SUMS syntax differs")
        digest, name = line.split("  ", 1)
        if name in observed_sums or name not in expected_sum_names or SHA256_RE.fullmatch(digest) is None:
            raise BatchError("one-case smoke SHA256SUMS coverage differs")
        observed_sums[name] = digest
    if set(observed_sums) != expected_sum_names:
        raise BatchError("one-case smoke SHA256SUMS exact coverage differs")
    for name, digest in observed_sums.items():
        if sha_file(root / name, f"one-case smoke SHA256SUMS {name}") != digest:
            raise BatchError(f"one-case smoke SHA256SUMS differs: {name}")
    case_binding_sha = sha_file(args.expanded, "one-case smoke case binding")
    expected_binding_keys = {"case_binding_sha256", "case_sha256", "fixture_sha256", "guard_set_sha256", "identity_file_sha256", "identity_self_sha256", "official_case_sha256", "package_content_sha256", "package_manifest_sha256", "policy_sha256", "preflight_sha256", "served_model_manifest_sha256", "worker_binary_sha256"}
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys or bindings.get("case_binding_sha256") != case_binding_sha or bindings.get("case_sha256") != TRUSTED_ONE_CASE_SHA256 or bindings.get("official_case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke bundle case binding/hash differs")
    if files["case-binding.json"]["sha256"] != case_binding_sha:
        raise BatchError("one-case smoke bundle file binding differs")
    if expanded.get("status") != "bound_one_case_smoke" or expanded.get("case_count") != 1 or expanded.get("canonical_case_sha256") != sha_bytes(canonical(cases)) or expanded.get("source_manifest_sha256") != TRUSTED_SOURCE_MANIFEST_SHA256 or expanded.get("official_case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke case-binding root differs")
    official = load(root / "official-case.json", "one-case smoke official case")
    official_case = official.get("case")
    if official.get("schema_version") != "ullm.aq4_p2_official_case.v1" or official.get("manifest_sha256") != TRUSTED_SOURCE_MANIFEST_SHA256 or not isinstance(official_case, dict) or official_case.get("case_id") != TRUSTED_ONE_CASE_ID or official_case.get("case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256 or case_hash(official_case) != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke trusted official case differs")
    index_cases = fixture_index.get("cases")
    fixture_entry = index_cases[0] if isinstance(index_cases, list) and len(index_cases) == 1 else None
    fixture_sha = sha_file(root / "fixture.json", "one-case smoke fixture")
    if fixture_index.get("subset") != "resident_one_case_smoke" or fixture_index.get("case_count") != 1 or fixture_index.get("expanded_manifest_sha256") != case_binding_sha or fixture_index.get("served_model_manifest_sha256") != identity.get("hash_binding", {}).get("served_model_manifest_sha256") or not isinstance(fixture_entry, dict) or fixture_entry.get("case_id") != TRUSTED_ONE_CASE_ID or fixture_entry.get("case_sha256") != TRUSTED_ONE_CASE_SHA256 or Path(fixture_entry.get("fixture_path", "")) != root / "fixture.json" or fixture_entry.get("fixture_sha256") != fixture_sha or bindings.get("fixture_sha256") != fixture_sha:
        raise BatchError("one-case smoke fixture index binding differs")
    identity_file_sha = sha_file(root / "identity.json", "one-case smoke identity")
    identity_self_sha = _identity_self_sha256(identity)
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("expanded_manifest_sha256") != case_binding_sha or identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != case_binding_sha or identity.get("identity_sha256") != identity_self_sha or bindings.get("identity_file_sha256") != identity_file_sha or bindings.get("identity_self_sha256") != identity_self_sha:
        raise BatchError("one-case smoke identity case binding differs")
    if bindings.get("preflight_sha256") != sha_file(root / "preflight.json", "one-case smoke preflight") or not isinstance(preflight.get("gpu_process_snapshot"), list):
        raise BatchError("one-case smoke preflight binding differs")
    if bindings.get("policy_sha256") != sha_file(root / "policy.json", "one-case smoke policy") or policy.get("schema_version") != "ullm.aq4_production_p2_threshold_policy.v1" or policy.get("status") != "bound":
        raise BatchError("one-case smoke policy binding differs")
    resident_identity = identity.get("resident_driver_identity", {})
    hash_binding = identity.get("hash_binding", {})
    for field in ("guard_set_sha256", "package_content_sha256", "package_manifest_sha256", "served_model_manifest_sha256", "worker_binary_sha256"):
        expected = resident_identity.get(field)
        if bindings.get(field) != expected or (field != "guard_set_sha256" and hash_binding.get(field) != expected):
            raise BatchError(f"one-case smoke {field} binding differs")
    if bindings.get("served_model_manifest_sha256") != files["served-model.json"]["sha256"] or bindings.get("package_manifest_sha256") != files["package-manifest.json"]["sha256"]:
        raise BatchError("one-case smoke served/package snapshot binding differs")
    expected_binary_sha256 = identity.get("resident_driver_identity", {}).get("binary_sha256")
    if not isinstance(expected_binary_sha256, str) or SHA256_RE.fullmatch(expected_binary_sha256) is None or files["resident-driver"]["sha256"] != expected_binary_sha256:
        raise BatchError("one-case smoke resident binary binding is invalid")
    launch = load(root / "launch-command.json", "one-case smoke launch command")
    expected_driver_argv = launch.get("resident_driver_argv")
    if (
        launch.get("schema_version") != "ullm.aq4_p2_resident_launch_command.v1"
        or not isinstance(expected_driver_argv, list)
        or expected_driver_argv[0:1] != [str(root / "resident-driver")]
    ):
        raise BatchError("one-case smoke resident driver argv binding is invalid")
    validate_driver_argv_schema(expected_driver_argv, identity)
    fake_ready = _run_fake_ready_handshake(fake_ready_path, args.timeout)
    session_id, driver_identity = validate_ready(fake_ready, identity, cases, expected_binary_sha256)
    prepared_plan = load(root / "dry-run.json", "one-case smoke prepared dry-run")
    prepared_validation = prepared_plan.get("validation")
    prepared_baseline = prepared_plan.get("baseline_identity")
    if (
        prepared_plan.get("case_count") != 1
        or prepared_plan.get("transaction_count") != WARMUP_RUNS + MEASURED_RUNS
        or prepared_plan.get("smoke_only") is not True
        or prepared_plan.get("promotion_eligible") is not False
        or not isinstance(prepared_validation, dict)
        or prepared_validation.get("fake_ready") != {"path": str(fake_ready_path), "sha256": files["fake-ready.json"]["sha256"]}
        or prepared_validation.get("resident_session_id") != session_id
        or prepared_validation.get("driver_identity") != driver_identity
        or not isinstance(prepared_baseline, dict)
        or prepared_baseline.get("identity_file") != {"path": str(root / "identity.json"), "sha256": identity_file_sha}
    ):
        raise BatchError("one-case smoke prepared dry-run identity/handshake binding differs")
    prepared_evidence = load(root / "runner-dry-run-evidence.json", "one-case smoke prepared runner evidence")
    if (
        prepared_evidence.get("schema_version") != "ullm.aq4_p2_resident_runner_subprocess_evidence.v1"
        or prepared_evidence.get("runner_subprocess_count") != 1
        or prepared_evidence.get("runner_source_sha256") != files["trusted-runner.py"]["sha256"]
        or prepared_evidence.get("plan") != {"path": str(root / "dry-run.json"), "sha256": observed_sums["dry-run.json"]}
    ):
        raise BatchError("one-case smoke prepared runner evidence binding differs")
    validator_path = args.trusted_validator
    validator = _run_bundle_validator(validator_path, args.trusted_validator_sha256, root, args.timeout)
    final_names = set(os.listdir(root_descriptor)) if root_descriptor is not None else {entry.name for entry in root.iterdir()}
    final_root_metadata = ACTIVE_FD_MAP.verify_binding(root_binding) if root_binding is not None else os.lstat(root)
    if final_names != ONE_CASE_ROOT_MEMBERS or _file_identity(final_root_metadata) != _file_identity(root_metadata):
        raise BatchError("one-case smoke bundle root changed during validation")
    for name, before in initial.items():
        member_path = root / name
        member_binding = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(member_path)
        current = (
            ACTIVE_FD_MAP.verify_binding(member_binding)
            if member_binding is not None
            else os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            if root_descriptor is not None
            else os.lstat(member_path)
        )
        if _file_identity(current) != before:
            raise BatchError(f"one-case smoke bundle member changed during validation: {name}")
    return bundle, {
        "mode": "validate_only",
        "root_contract": ONE_CASE_ROOT_CONTRACT,
        "bundle_root": {"path": str(root), "device": root_metadata.st_dev, "inode": root_metadata.st_ino},
        "members": member_inventory,
        "bundle": {"path": str(bundle_path), "sha256": sha_file(bundle_path, "one-case smoke bundle")},
        "fake_ready": {"path": str(fake_ready_path), "sha256": sha_file(fake_ready_path, "one-case smoke fake-ready")},
        "fake_driver_subprocess_count": 1,
        "driver_fake_handshake": "passed",
        "resident_session_id": session_id,
        "driver_identity": driver_identity,
        "resident_driver_argv": expected_driver_argv,
        "trusted_bundle_validator": validator,
    }


DRIVER_SECRET_MARKERS = (b"authorization:", b"bearer ", b"api_key", b"api-key", b"x-api-key")


def _tail_update(tail: bytearray, chunk: bytes) -> None:
    tail.extend(chunk)
    if len(tail) > MAX_DRIVER_TAIL_BYTES:
        del tail[:-MAX_DRIVER_TAIL_BYTES]


def _secret_scan(previous: bytes, chunk: bytes) -> tuple[bytes, bool]:
    combined = (previous + chunk).lower()
    detected = any(marker in combined for marker in DRIVER_SECRET_MARKERS)
    return combined[-64:], detected


class DriverAudit:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.stderr_capture_path = output_dir / ".resident-driver.stderr.incomplete"
        descriptor = os.open(
            self.stderr_capture_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        self.stderr_handle = os.fdopen(descriptor, "wb", buffering=0)
        self.spawned_unix_ns: int | None = None
        self.pid: int | None = None
        self.pgid: int | None = None
        self.stdout_event_count = 0
        self.stdout_records: list[dict[str, Any]] = []
        self.stdout_records_dropped = 0
        self.ready_received = False
        self.case_begin_count = 0
        self.warmup_completed = 0
        self.measured_completed = 0
        self.case_end_count = 0
        self.last_stage = "spawn"

    def mark_spawned(self, process: subprocess.Popen[bytes]) -> None:
        self.spawned_unix_ns = time.time_ns()
        self.pid = process.pid
        self.pgid = os.getpgid(process.pid)

    def record_stdout(self, record: dict[str, Any]) -> None:
        if len(self.stdout_records) == 64:
            self.stdout_records.pop(0)
            self.stdout_records_dropped += 1
        self.stdout_records.append(record)

    def close_parent_stderr(self) -> None:
        if not self.stderr_handle.closed:
            self.stderr_handle.flush()
            os.fsync(self.stderr_handle.fileno())
            self.stderr_handle.close()

    def finalize_stderr(self) -> dict[str, Any]:
        self.close_parent_stderr()
        digest = hashlib.sha256()
        total = 0
        tail = bytearray()
        overlap = b""
        secret_detected = False
        with self.stderr_capture_path.open("rb", buffering=0) as source:
            while chunk := source.read(DRIVER_IO_CHUNK_BYTES):
                digest.update(chunk)
                total += len(chunk)
                _tail_update(tail, chunk)
                overlap, detected = _secret_scan(overlap, chunk)
                secret_detected = secret_detected or detected
        metadata = self.stderr_capture_path.lstat()
        retained_path: Path | None = None
        retained_kind = "none"
        omission_reason: str | None = "empty" if total == 0 else None
        if total and not secret_detected and total <= MAX_DRIVER_STDERR_RETAIN_BYTES:
            retained_path = self.output_dir / "resident-driver.stderr.log"
            os.chmod(self.stderr_capture_path, 0o444)
            os.rename(self.stderr_capture_path, retained_path)
            retained_kind = "complete"
        elif total and not secret_detected:
            try:
                bytes(tail).decode("utf-8")
            except UnicodeError:
                omission_reason = "tail_not_utf8"
            else:
                retained_path = self.output_dir / "resident-driver.stderr.tail.log"
                atomic_write_bytes(retained_path, bytes(tail))
                retained_kind = "bounded_tail"
                omission_reason = "full_stream_exceeds_retention_bound"
        elif secret_detected:
            omission_reason = "secret_marker_detected"
        if self.stderr_capture_path.exists():
            self.stderr_capture_path.unlink()
        directory = os.open(self.output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {
            "bytes": total,
            "sha256": digest.hexdigest(),
            "source_identity": {"device": metadata.st_dev, "inode": metadata.st_ino, "mode": stat.S_IMODE(metadata.st_mode)},
            "retention_bound_bytes": MAX_DRIVER_STDERR_RETAIN_BYTES,
            "tail_bound_bytes": MAX_DRIVER_TAIL_BYTES,
            "retained_kind": retained_kind,
            "retained_path": str(retained_path) if retained_path is not None else None,
            "retained_sha256": sha_file(retained_path, "driver stderr retained evidence") if retained_path is not None else None,
            "secret_scan": {"performed": True, "markers": [marker.decode("ascii") for marker in DRIVER_SECRET_MARKERS], "detected": secret_detected},
            "omission_reason": omission_reason,
        }


def _send(process: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise BatchError("resident driver stdin is unavailable")
    process.stdin.write((json.dumps(message, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8"))
    process.stdin.flush()


def _recv(process: subprocess.Popen[bytes], timeout: float, audit: DriverAudit, stage: str) -> dict[str, Any]:
    if process.stdout is None:
        raise BatchError("resident driver stdout is unavailable")
    audit.last_stage = stage
    deadline = time.monotonic() + timeout
    digest = hashlib.sha256()
    tail = bytearray()
    captured = bytearray()
    total = 0
    ended = False
    timed_out = False
    secret_overlap = b""
    secret_detected = False
    while not ended:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if not ready:
            timed_out = True
            break
        chunk = process.stdout.readline(DRIVER_IO_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        _tail_update(tail, chunk)
        secret_overlap, detected = _secret_scan(secret_overlap, chunk)
        secret_detected = secret_detected or detected
        if len(captured) <= MAX_DRIVER_STDOUT_LINE_BYTES:
            remaining_capture = MAX_DRIVER_STDOUT_LINE_BYTES + 1 - len(captured)
            captured.extend(chunk[:remaining_capture])
        ended = chunk.endswith(b"\n")
    if total:
        audit.stdout_event_count += 1
    record = {
        "stage": stage,
        "bytes": total,
        "sha256": digest.hexdigest(),
        "line_complete": ended,
        "process_poll": process.poll(),
        "outcome": "pending",
        "tail_bytes": len(tail),
        "tail_sha256": sha_bytes(bytes(tail)),
        "tail_utf8": None,
        "tail_omission_reason": None,
    }
    if secret_detected:
        record["tail_omission_reason"] = "secret_marker_detected"
    else:
        try:
            record["tail_utf8"] = bytes(tail).decode("utf-8")
        except UnicodeError:
            record["tail_omission_reason"] = "tail_not_utf8"
    if timed_out:
        record["outcome"] = "timeout"
        audit.record_stdout(record)
        raise DriverProtocolError("timeout", stage, "resident driver response timed out")
    if total == 0:
        record["outcome"] = "eof"
        audit.record_stdout(record)
        raise DriverProtocolError("eof", stage, "resident driver exited before response")
    if not ended:
        record["outcome"] = "eof_mid_line"
        audit.record_stdout(record)
        raise DriverProtocolError("eof_mid_line", stage, "resident driver exited during response")
    if total > MAX_DRIVER_STDOUT_LINE_BYTES:
        record["outcome"] = "line_too_large"
        audit.record_stdout(record)
        raise DriverProtocolError("stdout_line_too_large", stage, "resident driver response exceeds line bound")
    line = bytes(captured)
    try:
        value = json.loads(line, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite driver number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        record["outcome"] = "invalid_json"
        audit.record_stdout(record)
        raise DriverProtocolError("invalid_json", stage, f"resident driver response JSON differs: {error}") from error
    if not isinstance(value, dict):
        record["outcome"] = "non_object"
        audit.record_stdout(record)
        raise DriverProtocolError("non_object", stage, "resident driver response is not an object")
    record["outcome"] = "json_object"
    audit.record_stdout(record)
    return value


def _process_group_alive(pgid: int | None) -> bool:
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_exit(pgid: int | None, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return True


def _driver_exit(returncode: int | None) -> dict[str, Any]:
    if returncode is None:
        return {"kind": "still_running", "exit_code": None, "signal": None, "oom_like_unconfirmed": False}
    if returncode < 0:
        signal_number = -returncode
        return {"kind": "signal", "exit_code": None, "signal": signal_number, "oom_like_unconfirmed": signal_number == signal.SIGKILL}
    return {"kind": "exit", "exit_code": returncode, "signal": None, "oom_like_unconfirmed": returncode == 137}


def _cleanup_driver(process: subprocess.Popen[bytes], audit: DriverAudit, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    record: dict[str, Any] = {
        "initial_poll": process.poll(),
        "shutdown_attempted": False,
        "shutdown_send_error": None,
        "stdin_closed": False,
        "wait_timed_out": False,
        "signals": [],
        "reaped": False,
        "final_returncode": None,
        "process_group_alive_final": None,
        "errors": [],
    }
    if process.poll() is None:
        record["shutdown_attempted"] = True
        try:
            _send(process, {"command": "shutdown", "schema_version": DRIVER_SCHEMA})
        except (BatchError, OSError) as error:
            record["shutdown_send_error"] = str(error)
    if process.stdin is not None:
        try:
            process.stdin.close()
            record["stdin_closed"] = True
        except OSError as error:
            record["errors"].append(f"stdin-close: {error}")
    try:
        process.wait(timeout=timeout)
        record["reaped"] = True
    except subprocess.TimeoutExpired:
        record["wait_timed_out"] = True
    grace = min(DRIVER_CLEANUP_GRACE_SECONDS, max(0.1, timeout))
    if process.poll() is None or _process_group_alive(audit.pgid):
        try:
            if audit.pgid is not None:
                os.killpg(audit.pgid, signal.SIGTERM)
                record["signals"].append("SIGTERM")
        except ProcessLookupError:
            pass
        except OSError as error:
            record["errors"].append(f"sigterm: {error}")
        try:
            process.wait(timeout=grace)
            record["reaped"] = True
        except subprocess.TimeoutExpired:
            pass
        _wait_process_group_exit(audit.pgid, grace)
    if process.poll() is None or _process_group_alive(audit.pgid):
        try:
            if audit.pgid is not None:
                os.killpg(audit.pgid, signal.SIGKILL)
                record["signals"].append("SIGKILL")
        except ProcessLookupError:
            pass
        except OSError as error:
            record["errors"].append(f"sigkill: {error}")
        try:
            process.wait(timeout=grace)
            record["reaped"] = True
        except subprocess.TimeoutExpired as error:
            record["errors"].append(f"final-wait: {error}")
        _wait_process_group_exit(audit.pgid, grace)
    record["final_returncode"] = process.poll()
    record["process_group_alive_final"] = _process_group_alive(audit.pgid)
    if process.stdout is not None:
        process.stdout.close()
    stderr = audit.finalize_stderr()
    record["passed"] = record["reaped"] is True and record["process_group_alive_final"] is False and not record["errors"]
    return record, stderr


def _driver_process_document(
    audit: DriverAudit,
    cleanup: dict[str, Any],
    stderr: dict[str, Any],
    lock_owner: dict[str, Any],
    error: BaseException | None,
) -> dict[str, Any]:
    try:
        lock_metadata = os.lstat(lock_owner["path"])
        lock_after = {
            "present": True,
            "device": lock_metadata.st_dev,
            "inode": lock_metadata.st_ino,
            "same_inode": lock_metadata.st_dev == lock_owner.get("device") and lock_metadata.st_ino == lock_owner.get("inode"),
        }
    except OSError as lock_error:
        lock_after = {"present": False, "device": None, "inode": None, "same_inode": False, "error_type": type(lock_error).__name__}
    failure = None
    if error is not None:
        failure = {
            "kind": error.kind if isinstance(error, DriverProtocolError) else type(error).__name__,
            "stage": error.stage if isinstance(error, DriverProtocolError) else audit.last_stage,
            "reason": str(error),
        }
    return {
        "schema_version": "ullm.aq4_p2_resident_driver_process.v1",
        "status": "failed" if error is not None else "complete",
        "spawn": {"captured_unix_ns": audit.spawned_unix_ns, "pid": audit.pid, "process_group_id": audit.pgid},
        "protocol": {
            "last_stage": audit.last_stage,
            "stdout_event_count": audit.stdout_event_count,
            "stdout_records": audit.stdout_records,
            "stdout_records_dropped": audit.stdout_records_dropped,
            "ready_received": audit.ready_received,
            "case_begin_count": audit.case_begin_count,
            "warmup_completed": audit.warmup_completed,
            "measured_completed": audit.measured_completed,
            "case_end_count": audit.case_end_count,
        },
        "failure": failure,
        "exit": _driver_exit(cleanup.get("final_returncode")),
        "stderr": stderr,
        "cleanup": cleanup,
        "lock": {"expected": {key: lock_owner.get(key) for key in ("path", "device", "inode")}, "after_driver": lock_after, "held_by_runner_during_capture": True},
        "gpu_owner": {"status": "not_probed", "reason": "runner_has_no_pinned_post_driver_gpu_owner_probe"},
        "secret_material_recorded": False,
    }


def _validate_ready_identity(value: Any, identity: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != READY_IDENTITY_KEYS:
        raise BatchError("resident driver identity fields differ")
    for field in ("binary_sha256", "worker_binary_sha256", "package_manifest_sha256", "package_content_sha256", "served_model_manifest_sha256", "guard_set_sha256"):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            raise BatchError(f"resident driver identity.{field} is invalid")
    if not isinstance(value["build_git_commit"], str) or GIT_SHA_RE.fullmatch(value["build_git_commit"]) is None:
        raise BatchError("resident driver identity.build_git_commit is invalid")
    if value["protocol"] != DRIVER_SCHEMA:
        raise BatchError("resident driver identity protocol differs")
    for field in ("model_id", "model_revision", "format_id", "implementation_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise BatchError(f"resident driver identity.{field} is invalid")
    runtime = value["runtime_device"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_DEVICE_KEYS:
        raise BatchError("resident driver runtime device fields differ")
    if type(runtime["runtime_device_index"]) is not int or runtime["runtime_device_index"] < 0:
        raise BatchError("resident driver runtime device index is invalid")
    if not isinstance(runtime["device_id"], (str, int)) or isinstance(runtime["device_id"], bool) or (isinstance(runtime["device_id"], str) and not runtime["device_id"]):
        raise BatchError("resident driver runtime device ID is invalid")
    for field in ("backend", "name", "architecture"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise BatchError(f"resident driver runtime device {field} is invalid")
    bound = identity.get("resident_driver_identity")
    if not isinstance(bound, dict) or set(bound) != READY_IDENTITY_KEYS:
        raise BatchError("identity file lacks resident driver identity")
    if bound != value:
        raise BatchError("resident driver identity differs from identity file")
    bound_hashes = identity.get("hash_binding", {})
    for field in ("package_manifest_sha256", "package_content_sha256", "served_model_manifest_sha256", "worker_binary_sha256"):
        if isinstance(bound_hashes, dict) and bound_hashes.get(field) != value[field]:
            raise BatchError(f"resident {field} identity differs")
    if identity.get("build_git_commit") not in (None, value["build_git_commit"]):
        raise BatchError("resident build commit identity differs")
    for case in cases:
        device = case.get("device")
        if not isinstance(device, dict):
            raise BatchError(f"case device identity is missing: {case.get('case_id')}")
        for field in RUNTIME_DEVICE_KEYS:
            expected = device.get(field)
            if field == "architecture" and runtime[field] == "gfx1201" and expected == "RDNA4":
                expected = "gfx1201"
            if field in device and expected != runtime[field]:
                raise BatchError(f"resident runtime device differs from case: {case['case_id']}")
    return value


def validate_ready(value: dict[str, Any], identity: dict[str, Any], cases: list[dict[str, Any]], expected_binary_sha256: str) -> tuple[str, dict[str, Any]]:
    if set(value) != {"event", "schema_version", "model_loads", "resident_session_id", "driver_identity"} or value.get("event") != "ready" or value.get("schema_version") != DRIVER_SCHEMA or type(value.get("model_loads")) is not int or value.get("model_loads") != 1 or not isinstance(value.get("resident_session_id"), str) or not value["resident_session_id"]:
        raise BatchError("resident driver did not prove one model load")
    ready_identity = _validate_ready_identity(value["driver_identity"], identity, cases)
    if ready_identity["binary_sha256"] != expected_binary_sha256:
        raise BatchError("resident driver ready self SHA differs from pre-spawn executable")
    return value["resident_session_id"], ready_identity


def validate_run(value: dict[str, Any], case: dict[str, Any], session_id: str) -> dict[str, Any]:
    required = {"event", "schema_version", "resident_session_id", "case_id", "run_index", "run_kind", "status", "elapsed_ms", "requested_m", "resolved_m", "actual_token_batch_width", "actual_request_batch_width", "timing", "audit", "state", "lifecycle", "reset", "resource", "terminal"}
    if set(value) != required or value.get("event") != "run_complete" or value.get("schema_version") != DRIVER_SCHEMA or value.get("resident_session_id") != session_id or value.get("case_id") != case["case_id"] or value.get("status") not in {"ok", "failed", "oom"}:
        raise BatchError(f"resident driver run identity/status differs: {case['case_id']}")
    if type(value.get("elapsed_ms")) not in {int, float} or value["elapsed_ms"] < 0:
        raise BatchError("resident driver elapsed_ms is invalid")
    reset = value.get("reset")
    terminal = value.get("terminal")
    if not isinstance(terminal, dict) or set(terminal) != {"reuse_forbidden", "reason_code", "oom", "hip_fault"} or type(terminal["reuse_forbidden"]) is not bool or type(terminal["oom"]) is not bool or type(terminal["hip_fault"]) is not bool or not isinstance(terminal["reason_code"], str):
        raise BatchError(f"resident driver terminal fields differ: {case['case_id']}")
    valid_resets = ({"attempted": 1, "complete": 1, "failed": 0}, {"attempted": 1, "complete": 0, "failed": 1}) if terminal["reuse_forbidden"] else ({"attempted": 1, "complete": 1, "failed": 0},)
    if not isinstance(reset, dict) or reset not in valid_resets:
        raise BatchError(f"resident driver reset is not complete: {case['case_id']}")
    if value["status"] == "oom" and (not terminal["oom"] or not terminal["reuse_forbidden"]):
        raise BatchError("resident driver OOM is not terminal")
    if terminal["hip_fault"] and not terminal["reuse_forbidden"]:
        raise BatchError("resident driver HIP fault is reusable")
    if value["status"] == "ok":
        audit = value.get("audit")
        resource = value.get("resource")
        if not isinstance(audit, dict) or audit.get("coverage_complete") is not True or not isinstance(audit.get("deterministic_digest_sha256"), str) or not isinstance(resource, dict) or not resource.get("samples") or not isinstance(resource.get("peak"), dict):
            raise BatchError(f"resident driver terminal audit/resource is incomplete: {case['case_id']}")
        if terminal["reuse_forbidden"] or value.get("requested_m") != case.get("prefill_requested_m") or value.get("resolved_m") != case.get("resolved_m") or value.get("actual_token_batch_width") != case.get("resolved_m") or value.get("actual_request_batch_width") != case.get("request_count"):
            raise BatchError(f"resident driver actual width differs: {case['case_id']}")
        for field in ("timing", "state", "lifecycle"):
            if not isinstance(value.get(field), dict):
                raise BatchError(f"resident driver {field} is incomplete: {case['case_id']}")
    return value


def make_case_raw(case: dict[str, Any], fixture_entry: dict[str, Any], identity_link: dict[str, str], policy_link: dict[str, str], run_id: str, baseline_kind: str, session_id: str, driver_identity: dict[str, Any], device_lock: dict[str, Any], runs: list[dict[str, Any]], failure_reason: str | None = None, live_preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    status = "ok" if not failure_reason and all(run["status"] == "ok" for run in runs) else "oom" if any(run["status"] == "oom" for run in runs) else "failed"
    terminal = {
        "audit_digests": [run.get("audit", {}).get("deterministic_digest_sha256") for run in runs if isinstance(run.get("audit"), dict)],
        "reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0}),
        "all_resets_complete": all(run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0} for run in runs),
    }
    return {
        "schema_version": "ullm.aq4_p2_resident_batch_raw.v1",
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "status": status,
        "immutable_status": status != "ok",
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": identity_link,
        },
        "resident": {"session_id": session_id, "model_loads": 1, "driver_identity": driver_identity, "case_reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0})},
        "device_lock": device_lock,
        "workload": {key: case.get(key) for key in ("scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "prefill_requested_m", "resolved_m", "request_count", "generated_tokens")},
        "schedule": {"warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS, "completed_runs": len(runs)},
        "runs": runs,
        "terminal": terminal,
        "failure_reason": failure_reason,
        "links": {"fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]}, "identity": identity_link, "policy": policy_link, **({"live_preflight": live_preflight} if live_preflight is not None else {})},
    }


def build_plan(cases: list[dict[str, Any]], expanded_path: Path, fixture_index_path: Path, run_id: str, baseline_kind: str, identity: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    token_sum = sum(int(case["prompt_tokens"]) for case in cases) * (WARMUP_RUNS + MEASURED_RUNS)
    return {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "scope": "full_model",
        "case_count": len(cases),
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "transaction_count": len(cases) * (WARMUP_RUNS + MEASURED_RUNS),
        "prompt_tokens_across_transactions": token_sum,
        "resident_model_loads": 1,
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": {"path": str(identity.get("_path", "")), "sha256": identity.get("_sha256")},
            "served_model_manifest_sha256": identity.get("hash_binding", {}).get("served_model_manifest_sha256"),
            "worker_binary_sha256": identity.get("hash_binding", {}).get("worker_binary_sha256"),
            "build_git_commit": identity.get("build_git_commit"),
        },
        "links": {"expanded": {"path": str(expanded_path), "sha256": sha_file(expanded_path, "expanded")}, "fixture_index": {"path": str(fixture_index_path), "sha256": sha_file(fixture_index_path, "fixture index")}, "policy": {"path": str(policy.get("_path", "")), "sha256": policy.get("_sha256")}},
    }


def validate_prepared_preflight_link(path: Path, value: dict[str, Any]) -> PreparedPreflightLink:
    if set(value) != PREPARED_PREFLIGHT_FIELDS:
        raise BatchError("prepared preflight fields differ")
    for field in PREPARED_PREFLIGHT_FIELDS - {"gpu_process_snapshot"}:
        if type(value.get(field)) is not int or value[field] < 0:
            raise BatchError(f"prepared preflight {field} is invalid")
    processes = value.get("gpu_process_snapshot")
    if not isinstance(processes, list):
        raise BatchError("prepared preflight process snapshot is invalid")
    for process in processes:
        if not isinstance(process, dict) or set(process) != {"pid", "process_name", "vram_bytes"} or type(process.get("pid")) is not int or process["pid"] < 0 or not isinstance(process.get("process_name"), str) or type(process.get("vram_bytes")) is not int or process["vram_bytes"] < 0:
            raise BatchError("prepared preflight process entry is invalid")
    resolved = path if ACTIVE_FD_MAP is not None and ACTIVE_FD_MAP.binding(path) is not None else path.resolve(strict=True)
    link: PreparedPreflightLink = {"path": str(resolved), "sha256": sha_file(path, "prepared preflight")}
    return require_prepared_preflight_link(link)


def require_prepared_preflight_link(value: dict[str, Any]) -> PreparedPreflightLink:
    if set(value) != {"path", "sha256"} or not isinstance(value.get("path"), str) or not Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts or not isinstance(value.get("sha256"), str) or SHA256_RE.fullmatch(value["sha256"]) is None:
        raise BatchError("prepared preflight link fields differ")
    return {"path": value["path"], "sha256": value["sha256"]}


def require_live_preflight_link(value: dict[str, Any]) -> LivePreflightLink:
    exact = {"path", "sha256", "device", "inode", "captured_unix_ns", "runtime_mapping", "lock", "vram"}
    if set(value) != exact or not isinstance(value.get("path"), str) or not Path(value["path"]).is_absolute() or not isinstance(value.get("sha256"), str) or SHA256_RE.fullmatch(value["sha256"]) is None:
        raise BatchError("live preflight link fields differ")
    if any(type(value.get(field)) is not int or value[field] < 0 for field in ("device", "inode", "captured_unix_ns")) or any(not isinstance(value.get(field), dict) for field in ("runtime_mapping", "lock", "vram")):
        raise BatchError("live preflight link identity fields differ")
    return {
        "path": value["path"], "sha256": value["sha256"], "device": value["device"],
        "inode": value["inode"], "captured_unix_ns": value["captured_unix_ns"],
        "runtime_mapping": value["runtime_mapping"], "lock": value["lock"], "vram": value["vram"],
    }


def validate_live_preflight(path: Path, args: argparse.Namespace, bundle: dict[str, Any]) -> LivePreflightLink:
    mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path, method="read")
    if mapped is None:
        _require_absolute_nonsymlink_path(path, "live preflight")
    raw, digest, before = read_regular(path, "live preflight", MAX_JSON_BYTES, absolute=True)
    if stat.S_IMODE(before.st_mode) != 0o444:
        raise BatchError("live preflight mode must be 0444")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite live preflight number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid live preflight: {error}") from error
    exact = {"schema_version", "status", "run_id", "captured_unix_ns", "prepared_preflight", "runtime_mapping", "services", "worker_pids", "compute_owners", "lock", "environment", "vram", "commands"}
    if not isinstance(value, dict) or set(value) != exact or value.get("schema_version") != "ullm.aq4_p2_resident_live_preflight.v1" or value.get("status") != "passed" or value.get("run_id") != args.run_id or type(value.get("captured_unix_ns")) is not int or value["captured_unix_ns"] <= 0:
        raise BatchError("live preflight exact schema/status/run differs")
    prepared = value.get("prepared_preflight")
    expected_prepared = {"path": str(args.bundle_root / "preflight.json"), "sha256": sha_file(args.bundle_root / "preflight.json", "prepared synthetic preflight"), "role": "synthetic_bundle_contract_only"}
    if prepared != expected_prepared:
        raise BatchError("live preflight does not explicitly replace the synthetic bundle preflight")
    mapping = value.get("runtime_mapping")
    expected_device = bundle.get("expected_runtime", {}).get("device", {})
    expected_mapping = {"runtime_device_index": expected_device.get("runtime_device_index"), "visible_token": "1", "amd_smi_index": 2, "bdf": "0000:47:00.0", "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55", "kfd_id": 51545, "node_id": 2}
    if mapping != expected_mapping:
        raise BatchError("live preflight runtime mapping differs")
    services = value.get("services")
    if services != [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}]:
        raise BatchError("live preflight service state differs")
    owners = value.get("compute_owners")
    if value.get("worker_pids") != [] or owners != {"amd_smi": [], "kfd": []}:
        raise BatchError("live preflight compute owners differ")
    lock = value.get("lock")
    if not isinstance(lock, dict) or set(lock) != {"path", "free", "device", "inode"} or lock.get("path") != str(args.lock_path) or lock.get("free") is not True or type(lock.get("device")) is not int or lock["device"] < 0 or type(lock.get("inode")) is not int or lock["inode"] < 0:
        raise BatchError("live preflight lock binding differs")
    expected_environment = bundle.get("expected_runtime", {}).get("environment", {}) | bundle.get("expected_runtime", {}).get("required_guards", {}) | {"ULLM_SERVED_MODEL_MANIFEST": "/etc/ullm/served-models/active.json", "ULLM_BUILD_GIT_COMMIT": bundle.get("resident_driver", {}).get("source_commit")}
    if value.get("environment") != expected_environment:
        raise BatchError("live preflight environment differs")
    vram = value.get("vram")
    if not isinstance(vram, dict) or set(vram) != {"total_bytes", "used_bytes", "free_bytes", "headroom_bytes"} or any(type(vram.get(name)) is not int or vram[name] < 0 for name in vram) or vram["total_bytes"] < 30_000_000_000 or vram["used_bytes"] != 0 or vram["free_bytes"] != vram["total_bytes"] or vram["headroom_bytes"] != vram["total_bytes"]:
        raise BatchError("live preflight VRAM/headroom differs")
    commands = value.get("commands")
    project_root = args.bundle_root.parents[5]
    expected_commands = {
        "sudo-n": (["/usr/bin/sudo", "-n", "-v"], 0),
        "service-ullm-openai.service": (["/usr/bin/systemctl", "show", "ullm-openai.service", "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], 0),
        "service-llama-qwen35-udq4.service": (["/usr/bin/systemctl", "show", "llama-qwen35-udq4.service", "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], 0),
        "old-worker": (["/usr/bin/pgrep", "-f", "-x", f"{project_root / 'target/reasoning-v2/release/ullm-aq4-worker'}.*"], 1),
        "amd-smi-list": (["/opt/rocm/bin/amd-smi", "list", "--json"], 0),
        "rocminfo": (["/usr/bin/rocminfo"], 0),
        "amd-smi-process": (["/opt/rocm/bin/amd-smi", "process", "--gpu", "2", "--general", "--json"], 0),
        "amd-smi-static-vram": (["/opt/rocm/bin/amd-smi", "static", "--gpu", "2", "--vram", "--json"], 0),
    }
    if not isinstance(commands, list) or len(commands) != len(expected_commands) or any(not isinstance(item, dict) for item in commands):
        raise BatchError("live preflight command evidence differs")
    observed_labels: set[str] = set()
    for item in commands:
        label = item.get("label")
        if set(item) != {"label", "argv", "exit_code", "stdout_sha256", "stderr_sha256", "captured_unix_ns"} or label not in expected_commands or label in observed_labels:
            raise BatchError("live preflight command evidence differs")
        expected_argv, expected_exit = expected_commands[label]
        if item.get("argv") != expected_argv or item.get("exit_code") != expected_exit or not isinstance(item.get("stdout_sha256"), str) or not SHA256_RE.fullmatch(item["stdout_sha256"]) or not isinstance(item.get("stderr_sha256"), str) or not SHA256_RE.fullmatch(item["stderr_sha256"]) or type(item.get("captured_unix_ns")) is not int or item["captured_unix_ns"] < 0:
            raise BatchError("live preflight command evidence differs")
        observed_labels.add(label)
    if observed_labels != set(expected_commands):
        raise BatchError("live preflight command evidence differs")
    after = ACTIVE_FD_MAP.verify_binding(mapped) if mapped is not None else os.lstat(path)
    if _file_identity(before) != _file_identity(after) or sha_file(path, "live preflight", absolute=True) != digest:
        raise BatchError("live preflight changed during validation")
    return require_live_preflight_link({"path": str(path), "sha256": digest, "device": before.st_dev, "inode": before.st_ino, "captured_unix_ns": value["captured_unix_ns"], "runtime_mapping": mapping, "lock": lock, "vram": vram})


def verify_live_preflight(path: Path, link: LivePreflightLink) -> None:
    mapped = None if ACTIVE_FD_MAP is None else ACTIVE_FD_MAP.binding(path, method="read")
    try:
        metadata = ACTIVE_FD_MAP.verify_binding(mapped) if mapped is not None else os.lstat(path)
    except OSError as error:
        raise BatchError(f"live preflight final metadata failed: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise BatchError("live preflight final file contract differs")
    if metadata.st_dev != link.get("device") or metadata.st_ino != link.get("inode"):
        raise BatchError("live preflight identity changed after validation")
    if sha_file(path, "live preflight final", absolute=True) != link.get("sha256"):
        raise BatchError("live preflight content changed after validation")


def roctx_marker_name(
    run_id: str,
    session_id: str,
    case_id: str,
    case_sha256: str,
    run_index: int,
    run_kind: str,
) -> str:
    for label, value in (
        ("run_id", run_id),
        ("session_id", session_id),
        ("case_id", case_id),
    ):
        if not isinstance(value, str) or not value or "/" in value or "=" in value:
            raise BatchError(f"ROCTx marker {label} is invalid")
    if not isinstance(case_sha256, str) or SHA256_RE.fullmatch(case_sha256) is None:
        raise BatchError("ROCTx marker case SHA-256 is invalid")
    expected_kind = "warmup" if run_index < WARMUP_RUNS else "measured"
    if (
        type(run_index) is not int
        or run_index < 0
        or run_index >= WARMUP_RUNS + MEASURED_RUNS
        or run_kind != expected_kind
    ):
        raise BatchError("ROCTx marker run index/kind is invalid")
    return (
        f"{ROCTX_MARKER_PREFIX}/run_id={run_id}/session_id={session_id}/"
        f"case_id={case_id}/case_sha256={case_sha256}/"
        f"run_index={run_index}/run_kind={run_kind}"
    )


def build_case_begin_command(
    case: dict[str, Any],
    fixture_entry: dict[str, Any],
    case_binding_link: dict[str, Any],
    identity_link: dict[str, Any],
    prepared_preflight_link: PreparedPreflightLink,
    policy_link: dict[str, Any],
) -> dict[str, Any]:
    sampling = case.get("sampling")
    control = case.get("control")
    if not isinstance(sampling, dict) or not isinstance(control, dict):
        raise BatchError(f"case sampling/control is missing: {case['case_id']}")
    preflight = require_prepared_preflight_link(prepared_preflight_link)
    return {
        "command": "case_begin", "schema_version": DRIVER_SCHEMA,
        "case_id": case["case_id"], "case_sha256": case["case_sha256"],
        "case_binding": case_binding_link, "identity": identity_link,
        "preflight": preflight, "policy": policy_link,
        "fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]},
        "execution": {
            "scope": case.get("scope"), "phase": case.get("phase"), "mode": case.get("mode"),
            "prompt_tokens": case.get("prompt_tokens"), "cached_prefix_tokens": case.get("cached_prefix_tokens"),
            "context_tokens": case.get("context_tokens"), "generated_tokens": case.get("generated_tokens"),
            "request_count": case.get("request_count"), "requested_m": case.get("prefill_requested_m"),
            "resolved_m": case.get("resolved_m"), "sampling": sampling, "control": control,
        },
    }


def execute_resident_run(
    process: subprocess.Popen[bytes],
    audit: DriverAudit,
    case: dict[str, Any],
    session_id: str,
    run_index: int,
    run_kind: str,
    timeout: float,
    roctx: RoctxRangeRecorder | None,
    run_id: str,
) -> dict[str, Any]:
    manager: Any = nullcontext()
    if roctx is not None:
        marker = roctx_marker_name(
            run_id,
            session_id,
            case["case_id"],
            case["case_sha256"],
            run_index,
            run_kind,
        )
        manager = roctx.range(marker, run_index, run_kind)
    with manager:
        _send(
            process,
            {
                "command": "run",
                "schema_version": DRIVER_SCHEMA,
                "case_id": case["case_id"],
                "run_index": run_index,
                "run_kind": run_kind,
            },
        )
        value = validate_run(
            _recv(process, timeout, audit, f"run:{case['case_id']}:{run_index}"),
            case,
            session_id,
        )
        if value["run_index"] != run_index or value["run_kind"] != run_kind:
            raise BatchError(f"resident driver run order differs: {case['case_id']}")
        if run_kind == "warmup":
            audit.warmup_completed += 1
        else:
            audit.measured_completed += 1
        return value


def _run_batch(args: argparse.Namespace) -> int:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise BatchError("--timeout must be a finite positive number")
    if args.profile_roctx_ranges:
        if not args.one_case_smoke or args.dry_run:
            raise BatchError("--profile-roctx-ranges requires an actual --one-case-smoke run")
        if args.roctx_library is None or args.roctx_library_sha256 is None:
            raise BatchError(
                "--roctx-library and --roctx-library-sha256 are required with profiling ranges"
            )
    elif args.roctx_library is not None or args.roctx_library_sha256 is not None:
        raise BatchError("ROCTx library options require --profile-roctx-ranges")
    if not args.one_case_smoke and (args.bundle_root is not None or args.trusted_validator is not None or args.trusted_validator_sha256 is not None or args.live_preflight is not None):
        raise BatchError("--bundle-root/--trusted-validator require --one-case-smoke")
    if args.one_case_smoke and args.bundle_root is None:
        raise BatchError("--bundle-root is required with --one-case-smoke")
    if args.one_case_smoke and (args.trusted_validator is None or args.trusted_validator_sha256 is None):
        raise BatchError("--trusted-validator and --trusted-validator-sha256 are required with --one-case-smoke")
    if args.one_case_smoke and args.dry_run and args.live_preflight is not None:
        raise BatchError("--live-preflight is forbidden for dry-run")
    if args.one_case_smoke and not args.dry_run and args.live_preflight is None:
        raise BatchError("--live-preflight is required for actual one-case smoke")
    expanded = load(args.expanded, "expanded")
    fixture_index = load(args.fixture_index, "fixture index")
    identity = load(args.identity, "identity")
    preflight = load(args.preflight, "preflight")
    policy = load(args.policy, "policy")
    normalize_fixture_paths(fixture_index)
    expanded_link = {"path": str(args.expanded.resolve()), "sha256": sha_file(args.expanded, "expanded")}
    identity_link = {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity")}
    prepared_preflight_link = validate_prepared_preflight_link(args.preflight, preflight)
    policy_link = {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy")}
    identity["_path"], identity["_sha256"] = str(args.identity.resolve()), identity_link["sha256"]
    policy["_path"], policy["_sha256"] = str(args.policy.resolve()), policy_link["sha256"]
    if args.baseline_kind not in {"active-production", "p3-current-head"}:
        raise BatchError("baseline kind must identify one immutable build/run")
    cases = select_target_cases(expanded, fixture_index, one_case_smoke=args.one_case_smoke)
    smoke_validation = None
    smoke_bundle = None
    if args.one_case_smoke:
        smoke_bundle, smoke_validation = validate_one_case_smoke_bundle(args, expanded, fixture_index, identity, preflight, policy, cases)
    live_preflight_link: LivePreflightLink | None = None
    if args.one_case_smoke and not args.dry_run:
        live_preflight_link = validate_live_preflight(args.live_preflight, args, smoke_bundle)
        smoke_validation["live_preflight"] = live_preflight_link
    plan = build_plan(cases, args.expanded, args.fixture_index, args.run_id, args.baseline_kind, identity, policy)
    if ACTIVE_FD_MAP is not None:
        plan["execution_closure"] = ACTIVE_FD_MAP.evidence()
    if args.one_case_smoke:
        plan.update({"execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False, "validation": smoke_validation})
    if args.dry_run:
        atomic_write(args.output_dir / "resident-batch.plan.json", plan)
        return 0
    if not args.driver_command:
        raise BatchError("--driver-command is required unless --dry-run is set")
    expected_driver_argv = smoke_validation["resident_driver_argv"] if smoke_validation is not None else None
    driver_executable = validate_driver_command(args.driver_command, identity, expected_argv=expected_driver_argv)
    effective_driver_command = list(args.driver_command)
    if ACTIVE_FD_MAP is not None:
        effective_driver_command[0] = effective_fd_path(
            Path(args.driver_command[0]), method="exec", role="resident_driver"
        )
        effective_driver_command[2] = effective_fd_path(
            Path(args.driver_command[2]), method="read", role="served_manifest"
        )
    roctx = (
        RoctxRangeRecorder.load(args.roctx_library, args.roctx_library_sha256)
        if args.profile_roctx_ranges
        else None
    )
    if live_preflight_link is not None:
        verify_live_preflight(args.live_preflight, live_preflight_link)
    completed_cases = 0
    expected_lock_identity = None if live_preflight_link is None else {
        "device": live_preflight_link["lock"]["device"],
        "inode": live_preflight_link["lock"]["inode"],
    }
    with acquire_device_lock(
        args.lock_path,
        args.run_id,
        driver_executable,
        expected_identity=expected_lock_identity,
    ) as lock_owner:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write(args.output_dir / "resident-batch.lock-owner.json", lock_owner)
        process: subprocess.Popen[bytes] | None = None
        audit = DriverAudit(args.output_dir)
        driver_error: BaseException | None = None
        try:
            process = subprocess.Popen(
                effective_driver_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=audit.stderr_handle,
                text=False,
                shell=False,
                bufsize=0,
                start_new_session=True,
                **fd_child_options(),
            )
            audit.mark_spawned(process)
            session_id, driver_identity = validate_ready(
                _recv(process, args.timeout, audit, "ready"),
                identity,
                cases,
                driver_executable["sha256"],
            )
            audit.ready_received = True
            by_id = {entry["case_id"]: entry for entry in fixture_index["cases"]}
            for case in cases:
                if live_preflight_link is not None:
                    verify_live_preflight(args.live_preflight, live_preflight_link)
                fixture_entry = by_id[case["case_id"]]
                _send(process, build_case_begin_command(
                    case, fixture_entry, expanded_link, identity_link,
                    prepared_preflight_link, policy_link,
                ))
                begin = _recv(process, args.timeout, audit, f"case_begin:{case['case_id']}")
                if set(begin) != {"event", "schema_version", "resident_session_id", "case_id", "requested_m", "resolved_m", "baseline_clean"} or begin.get("event") != "case_ready" or begin.get("schema_version") != DRIVER_SCHEMA or begin.get("resident_session_id") != session_id or begin.get("case_id") != case["case_id"] or begin.get("requested_m") != case["prefill_requested_m"] or begin.get("resolved_m") != case["resolved_m"] or begin.get("baseline_clean") is not True:
                    raise BatchError(f"resident driver case begin failed: {case['case_id']}")
                audit.case_begin_count += 1
                runs: list[dict[str, Any]] = []
                reuse_forbidden = False
                for run_index in range(WARMUP_RUNS + MEASURED_RUNS):
                    run_kind = "warmup" if run_index < WARMUP_RUNS else "measured"
                    value = execute_resident_run(
                        process,
                        audit,
                        case,
                        session_id,
                        run_index,
                        run_kind,
                        args.timeout,
                        roctx,
                        args.run_id,
                    )
                    runs.append(value)
                    if value["terminal"]["reuse_forbidden"] or value["status"] != "ok":
                        reuse_forbidden = value["terminal"]["reuse_forbidden"]
                        break
                if not reuse_forbidden:
                    _send(process, {"command": "case_end", "schema_version": DRIVER_SCHEMA, "case_id": case["case_id"]})
                    end = _recv(process, args.timeout, audit, f"case_end:{case['case_id']}")
                    case_failed = any(item["status"] != "ok" for item in runs)
                    expected_release = {"commit": int(not case_failed), "discard": int(case_failed), "reset": 1, "baseline_restored": True}
                    if set(end) != {"event", "schema_version", "resident_session_id", "case_id", "release"} or end.get("event") != "case_complete" or end.get("schema_version") != DRIVER_SCHEMA or end.get("resident_session_id") != session_id or end.get("case_id") != case["case_id"] or end.get("release") != expected_release:
                        raise BatchError(f"resident driver case end failed: {case['case_id']}")
                    audit.case_end_count += 1
                failure_reason = "resident_driver_oom" if any(item["status"] == "oom" for item in runs) else next((item["terminal"]["reason_code"] for item in runs if item["status"] != "ok"), None)
                raw = make_case_raw(case, fixture_entry, identity_link, policy_link, args.run_id, args.baseline_kind, session_id, driver_identity, lock_owner, runs, failure_reason, live_preflight=live_preflight_link)
                if args.one_case_smoke:
                    raw.update({"execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False})
                atomic_write(args.output_dir / f"{case['case_id']}.raw.json", raw)
                completed_cases += 1
                if reuse_forbidden:
                    raise BatchError(f"resident driver became non-reusable at {case['case_id']}; remaining cases were not executed")
        except BaseException as error:
            driver_error = error
            raise
        finally:
            if roctx is not None:
                roctx.close_active()
            cleanup_error: BaseException | None = None
            if process is not None:
                try:
                    cleanup, stderr = _cleanup_driver(process, audit, args.timeout)
                    if not cleanup["passed"]:
                        cleanup_error = BatchError("resident driver or descendant cleanup failed")
                except BaseException as error:
                    cleanup_error = error
                    cleanup = {
                        "initial_poll": process.poll(),
                        "shutdown_attempted": False,
                        "shutdown_send_error": None,
                        "stdin_closed": False,
                        "wait_timed_out": False,
                        "signals": [],
                        "reaped": process.poll() is not None,
                        "final_returncode": process.poll(),
                        "process_group_alive_final": _process_group_alive(audit.pgid),
                        "errors": [f"cleanup-exception: {type(error).__name__}: {error}"],
                        "passed": False,
                    }
                    try:
                        stderr = audit.finalize_stderr()
                    except BaseException as stderr_error:
                        stderr = {
                            "bytes": None,
                            "sha256": None,
                            "retained_kind": "unavailable",
                            "omission_reason": f"finalization_failed:{type(stderr_error).__name__}",
                        }
            else:
                cleanup = {
                    "initial_poll": None,
                    "shutdown_attempted": False,
                    "shutdown_send_error": None,
                    "stdin_closed": False,
                    "wait_timed_out": False,
                    "signals": [],
                    "reaped": False,
                    "final_returncode": None,
                    "process_group_alive_final": False,
                    "errors": ["driver-spawn-failed"],
                    "passed": False,
                }
                cleanup_error = BatchError("resident driver spawn failed")
                stderr = audit.finalize_stderr()
            effective_error = driver_error if driver_error is not None else cleanup_error
            atomic_write(
                args.output_dir
                / ("resident-batch.failure.json" if effective_error is not None else "resident-batch.driver-process.json"),
                _driver_process_document(audit, cleanup, stderr, lock_owner, effective_error),
                mode=0o444,
            )
            if driver_error is None and cleanup_error is not None:
                raise cleanup_error
        if live_preflight_link is not None:
            verify_live_preflight(args.live_preflight, live_preflight_link)
        if roctx is not None:
            atomic_write(
                args.output_dir / "resident-batch.roctx-ranges.json",
                roctx.evidence(),
            )
        atomic_write(args.output_dir / "resident-batch.summary.json", {**plan, "status": "complete", "completed_cases": completed_cases, "device_lock": lock_owner})
    return 0


def run_batch(args: argparse.Namespace) -> int:
    global ACTIVE_FD_MAP
    if ACTIVE_FD_MAP is not None:
        raise BatchError("pinned FD map context is already active")
    pinned_map = PinnedFdMap.from_environment(required=args.profile_roctx_ranges)
    ACTIVE_FD_MAP = pinned_map
    try:
        if pinned_map is not None:
            raw_cli = getattr(args, "_logical_cli_argv", None)
            if not isinstance(raw_cli, list) or any(not isinstance(item, str) for item in raw_cli):
                raise BatchError("logical runner argv is unavailable for FD-map verification")
            logical_argv = [
                pinned_map.role("python_interpreter", method="exec")["logical_path"],
                pinned_map.role("resident_runner", method="exec")["logical_path"],
                *raw_cli,
            ]
            if sha_bytes(canonical(logical_argv)) != pinned_map.value["logical_argv_sha256"]:
                raise BatchError("logical runner argv differs from the pinned FD map")
            required_roles = {
                "python_interpreter", "resident_runner", "case_binding",
                "fixture_index", "identity", "prepared_preflight", "policy",
            }
            if args.one_case_smoke:
                required_roles |= {"trusted_validator", "bundle_root"}
            if not args.dry_run:
                required_roles |= {"resident_driver", "device_lock"}
            if args.profile_roctx_ranges:
                required_roles |= {"roctx_library", "live_preflight", "served_manifest"}
            if not required_roles <= set(pinned_map.roles):
                raise BatchError("pinned FD map required role coverage differs")
        return _run_batch(args)
    finally:
        try:
            if pinned_map is not None:
                pinned_map.verify_data_guards()
        finally:
            ACTIVE_FD_MAP = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--fixture-index", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-kind", choices=("active-production", "p3-current-head"), required=True)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--one-case-smoke", action="store_true", help="run the exact bundle-v3 one-case smoke; never promotion eligible")
    parser.add_argument("--bundle-root", type=Path, help="absolute complete 791a20c bundle root; required by --one-case-smoke")
    parser.add_argument("--trusted-validator", type=Path, help="trusted bundle validator Python source required by --one-case-smoke")
    parser.add_argument("--trusted-validator-sha256", help="expected lowercase SHA-256 of --trusted-validator")
    parser.add_argument("--live-preflight", type=Path, help="immutable live gate sidecar required for actual one-case smoke and forbidden for dry-run")
    parser.add_argument(
        "--profile-roctx-ranges",
        action="store_true",
        help="emit exact 12 diagnostic ROCTx run ranges; one-case actual runs only",
    )
    parser.add_argument("--roctx-library", type=Path)
    parser.add_argument("--roctx-library-sha256")
    parser.add_argument("--driver-command", nargs=argparse.REMAINDER, help="exact resident driver argv; this option and its value must be last")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._logical_cli_argv = raw_argv
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_batch(args)
    except (BatchError, OSError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
