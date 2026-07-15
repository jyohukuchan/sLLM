#!/usr/bin/env python3
"""Capture and split one marked AQ4 resident diagnostic rocprof session."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA256_RE = re.compile(r"[0-9a-f]{64}")
HELPER_MAX_BYTES = 2 * 1024 * 1024
PRODUCER_PATH = ROOT / "tools/build-aq4-p3-selection-raw.py"
PRODUCER_SHA256 = "ce31daba6737a64efd2db3b897bcbef56289052978e7b3be544f89d82b91da52"
SELECTOR_PATH = ROOT / "tools/select-aq4-p3-candidate.py"
SELECTOR_SHA256 = "4a510c7351131072ed368e2ac8fffeb2daf10488edef94c37fe5dbcb729e9739"
PROFILE_HELPER_PATH = ROOT / "tools/profile-aq4-p2-family-exclusive.py"
PROFILE_HELPER_SHA256 = "ef26005a364511ab8d0f7ca2fa46ad2108cac083d0a2a24721f6cef577e16c92"
FD_MAP_SCHEMA = "ullm.aq4_p3_inherited_fd_map.v1"
FD_MAP_ENV = "ULLM_AQ4_PINNED_FD_MAP"
FD_MAP_MAX_BYTES = 1024 * 1024


class CaptureError(ValueError):
    pass


def local_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class PinnedPythonHelper:
    def __init__(self, path: Path, sha256: str, descriptor: int, identity: tuple[int, ...], data: bytes) -> None:
        self.path = path
        self.sha256 = sha256
        self.descriptor = descriptor
        self.identity = identity
        self.data = data

    @classmethod
    def open(cls, path: Path, expected_sha256: str) -> "PinnedPythonHelper":
        if not path.is_absolute() or ".." in path.parts or HELPER_SHA256_RE.fullmatch(expected_sha256) is None:
            raise CaptureError("capture helper binding is invalid")
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CaptureError(f"capture helper path contains a symlink: {current}")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > HELPER_MAX_BYTES:
            raise CaptureError("capture helper must be a bounded single-link regular file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if local_identity(opened) != local_identity(metadata):
                raise CaptureError("capture helper changed while opening")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                size += len(chunk)
                if size > HELPER_MAX_BYTES:
                    raise CaptureError("capture helper exceeds the input limit")
                chunks.append(chunk)
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256 or local_identity(os.fstat(descriptor)) != local_identity(metadata):
                raise CaptureError("capture helper SHA-256 or identity differs")
            return cls(path, expected_sha256, descriptor, local_identity(metadata), b"".join(chunks))
        except Exception:
            os.close(descriptor)
            raise

    def load(self, name: str, *, injected_modules: dict[str, Any] | None = None) -> Any:
        module = types.ModuleType(name)
        module.__file__ = str(self.path)
        module.__package__ = ""
        if injected_modules is not None:
            if set(injected_modules) != {"selector", "profiler"} or any(not isinstance(item, types.ModuleType) for item in injected_modules.values()):
                raise CaptureError("capture helper module injection differs")
            module.__dict__["_ULLM_VERIFIED_MODULES"] = dict(injected_modules)
        sys.modules[name] = module
        try:
            code = compile(self.data, str(self.path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        finally:
            sys.modules.pop(name, None)
        return module

    def verify(self) -> None:
        if local_identity(self.path.lstat()) != self.identity or local_identity(os.fstat(self.descriptor)) != self.identity:
            raise CaptureError("capture helper identity changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(self.descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise CaptureError("capture helper bytes changed")

    def evidence(self, role: str) -> dict[str, Any]:
        return {"role": role, "path": str(self.path), "identity": list(self.identity), "sha256": self.sha256}


PRODUCER_HELPER = PinnedPythonHelper.open(PRODUCER_PATH, PRODUCER_SHA256)
SELECTOR_HELPER = PinnedPythonHelper.open(SELECTOR_PATH, SELECTOR_SHA256)
PROFILE_HELPER = PinnedPythonHelper.open(PROFILE_HELPER_PATH, PROFILE_HELPER_SHA256)
SELECTOR = SELECTOR_HELPER.load("aq4_p3_selector_for_diagnostic_capture")
PROFILER = PROFILE_HELPER.load("aq4_p2_profiler_for_diagnostic_capture")
PRODUCER = PRODUCER_HELPER.load(
    "aq4_p3_producer_for_diagnostic_capture",
    injected_modules={"selector": SELECTOR, "profiler": PROFILER},
)


def capture_helper_contract() -> list[dict[str, Any]]:
    return [
        PRODUCER_HELPER.evidence("selection_raw_producer"),
        SELECTOR_HELPER.evidence("candidate_selector"),
        PROFILE_HELPER.evidence("profile_family_classifier"),
    ]


def verify_capture_helpers() -> None:
    for helper in (PRODUCER_HELPER, SELECTOR_HELPER, PROFILE_HELPER):
        helper.verify()


SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_capture.v1"
TARGET_SCHEMA = "ullm.aq4_p3_profile_target_command.v1"
FAILURE_SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_failure.v2"
READY_CANDIDATE_AUDIT_SCHEMA = "ullm.aq4_p2_ready_candidate_audit.v1"
READY_CANDIDATE_CAPTURE_SCHEMA = "ullm.aq4_p3_ready_candidate_capture.v1"
READY_CANDIDATE_MARKER_PREFIX = b"ULLM_AQ4_READY_CANDIDATE_AUDIT_V1 "
MAX_READY_CANDIDATE_MARKER_BYTES = 16 * 1024
MARKER_PREFIX = "ullm.aq4_p2.run.v1"
MARKER_CLOCK = "rocprofv3_monotonic_ns"
MAX_ROWS = 500_000
MARKER_KEYS = {
    "run_id", "session_id", "case_id", "case_sha256", "run_index", "run_kind"
}
MEMORY_COPY_KINDS = {
    "d2h", "h2d", "d2d", "h2h", "peer", "peertopeer", "hosttodevice",
    "devicetohost", "devicetodevice", "hosttohost",
}


class SymlinkIdentity(NamedTuple):
    path: Path
    identity: tuple[int, ...]
    target: str


class PinnedProfiler:
    def __init__(
        self,
        invocation: Path,
        resolved: Path,
        expected_sha256: str,
        descriptor: int,
        identity: tuple[int, ...],
        symlinks: tuple[SymlinkIdentity, ...],
    ) -> None:
        self.invocation = invocation
        self.resolved = resolved
        self.sha256 = expected_sha256
        self.descriptor = descriptor
        self.identity = identity
        self.symlinks = symlinks

    @property
    def fd_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    @classmethod
    def open(cls, invocation: Path, expected_sha256: str) -> "PinnedProfiler":
        if not invocation.is_absolute() or ".." in invocation.parts:
            raise CaptureError("profiler path must be absolute without parent traversal")
        if (
            not isinstance(expected_sha256, str)
            or PRODUCER.SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise CaptureError("profiler expected SHA-256 is invalid")
        symlinks: list[SymlinkIdentity] = []
        current = Path(invocation.anchor)
        for part in invocation.parts[1:]:
            current /= part
            try:
                metadata = current.lstat()
            except OSError as error:
                raise CaptureError(
                    f"profiler invocation component is unavailable: {current}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                symlinks.append(
                    SymlinkIdentity(
                        current, PROFILER._identity(metadata), os.readlink(current)
                    )
                )
        try:
            resolved = invocation.resolve(strict=True)
        except OSError as error:
            raise CaptureError(f"profiler path resolution failed: {error}") from error
        metadata = resolved.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o111 == 0
        ):
            raise CaptureError("resolved profiler must be a single-link executable regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        try:
            opened = os.fstat(descriptor)
            if PROFILER._identity(opened) != PROFILER._identity(metadata):
                raise CaptureError("profiler changed while opening")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise CaptureError("profiler SHA-256 differs")
            if PROFILER._identity(os.fstat(descriptor)) != PROFILER._identity(metadata):
                raise CaptureError("profiler changed while hashing")
            value = cls(
                invocation,
                resolved,
                expected_sha256,
                descriptor,
                PROFILER._identity(metadata),
                tuple(symlinks),
            )
            value.verify()
            return value
        except Exception:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        try:
            for item in self.symlinks:
                current = item.path.lstat()
                if (
                    PROFILER._identity(current) != item.identity
                    or not stat.S_ISLNK(current.st_mode)
                    or os.readlink(item.path) != item.target
                ):
                    raise CaptureError(
                        f"profiler invocation symlink changed: {item.path}"
                    )
            if self.invocation.resolve(strict=True) != self.resolved:
                raise CaptureError("profiler resolved target changed")
            current = self.resolved.lstat()
            opened = os.fstat(self.descriptor)
        except OSError as error:
            raise CaptureError(f"profiler binding is unavailable: {error}") from error
        if (
            PROFILER._identity(current) != self.identity
            or PROFILER._identity(opened) != self.identity
        ):
            raise CaptureError("profiler inode identity changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(self.descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise CaptureError("profiler bytes changed")

    def evidence(self) -> dict[str, Any]:
        return {
            "tool": "rocprofv3",
            "invocation_path": str(self.invocation),
            "resolved_path": str(self.resolved),
            "executable_sha256": self.sha256,
            "resolved_identity": list(self.identity),
            "symlink_chain": [
                {"path": str(item.path), "identity": list(item.identity), "target": item.target}
                for item in self.symlinks
            ],
        }

    def close(self) -> None:
        os.close(self.descriptor)


class PinnedTargetManifest:
    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        descriptor: int,
        identity: tuple[int, ...],
    ) -> None:
        self.path = path
        self.sha256 = expected_sha256
        self.descriptor = descriptor
        self.identity = identity

    @classmethod
    def open(cls, path: Path, expected_sha256: str) -> "PinnedTargetManifest":
        if not path.is_absolute() or ".." in path.parts:
            raise CaptureError(
                "target command manifest path must be absolute without parent traversal"
            )
        if (
            not isinstance(expected_sha256, str)
            or PRODUCER.SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise CaptureError("target command manifest expected SHA-256 is invalid")
        try:
            resolved = PROFILER.canonical_path(path, "target command manifest")
            metadata = resolved.lstat()
        except (OSError, PROFILER.ProfileError) as error:
            raise CaptureError(f"target command manifest path is invalid: {error}") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CaptureError(
                "target command manifest must be a single-link regular file"
            )
        if metadata.st_size > PRODUCER.MAX_INPUT_BYTES:
            raise CaptureError("target command manifest exceeds the input limit")
        descriptor = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if PROFILER._identity(opened) != PROFILER._identity(metadata):
                raise CaptureError("target command manifest changed while opening")
            pinned = cls(
                resolved,
                expected_sha256,
                descriptor,
                PROFILER._identity(metadata),
            )
            pinned.verify()
            return pinned
        except Exception:
            os.close(descriptor)
            raise

    def read_verified(self) -> bytes:
        try:
            current = self.path.lstat()
            opened = os.fstat(self.descriptor)
        except OSError as error:
            raise CaptureError(
                f"target command manifest binding is unavailable: {error}"
            ) from error
        if (
            PROFILER._identity(current) != self.identity
            or PROFILER._identity(opened) != self.identity
        ):
            raise CaptureError("target command manifest inode identity changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(self.descriptor, 1024 * 1024):
            size += len(chunk)
            if size > PRODUCER.MAX_INPUT_BYTES:
                raise CaptureError("target command manifest exceeds the input limit")
            digest.update(chunk)
            chunks.append(chunk)
        if digest.hexdigest() != self.sha256:
            raise CaptureError("target command manifest file SHA-256 differs")
        if PROFILER._identity(os.fstat(self.descriptor)) != self.identity:
            raise CaptureError("target command manifest changed while hashing")
        return b"".join(chunks)

    def verify(self) -> None:
        data = self.read_verified()
        snapshot = PRODUCER.Snapshot(
            self.path,
            PRODUCER.file_identity(os.fstat(self.descriptor)),
            self.sha256,
            data,
        )
        value = PRODUCER.parse_json(snapshot, "target command manifest")
        validate_target_manifest_root(value)

    def close(self) -> None:
        os.close(self.descriptor)


class PinnedTargetFile:
    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        descriptor: int,
        identity: tuple[int, ...],
        argument_index: int | None,
        binding: dict[str, Any],
    ) -> None:
        self.path = path
        self.sha256 = expected_sha256
        self.descriptor = descriptor
        self.identity = identity
        self.argument_index = argument_index
        self.binding = binding

    @property
    def fd_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    @classmethod
    def open(
        cls,
        path: Path,
        expected_sha256: str,
        argument_index: int | None,
        *,
        require_executable: bool,
        binding: dict[str, Any],
    ) -> "PinnedTargetFile":
        if not path.is_absolute() or ".." in path.parts or PRODUCER.SHA256_RE.fullmatch(expected_sha256) is None:
            raise CaptureError("target input file binding is invalid")
        try:
            resolved = PROFILER.canonical_path(path, "target input file")
            metadata = resolved.lstat()
        except (OSError, PROFILER.ProfileError) as error:
            raise CaptureError(f"target input file path is invalid: {error}") from error
        if resolved != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CaptureError("target input file must be a canonical single-link regular file")
        if require_executable and stat.S_IMODE(metadata.st_mode) & 0o111 == 0:
            raise CaptureError("target input executable permission differs")
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            identity = PROFILER._identity(metadata)
            if PROFILER._identity(opened) != identity:
                raise CaptureError("target input file changed while opening")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256 or PROFILER._identity(os.fstat(descriptor)) != identity:
                raise CaptureError("target input file SHA-256 or identity differs")
            return cls(
                resolved,
                expected_sha256,
                descriptor,
                identity,
                argument_index,
                binding,
            )
        except Exception:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        try:
            current = self.path.lstat()
            opened = os.fstat(self.descriptor)
        except OSError as error:
            raise CaptureError(f"target input file binding is unavailable: {error}") from error
        if PROFILER._identity(current) != self.identity or PROFILER._identity(opened) != self.identity:
            raise CaptureError("target input file identity changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(self.descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise CaptureError("target input file bytes changed")

    def close(self) -> None:
        os.close(self.descriptor)


class PinnedRuntimePath:
    def __init__(self, value: dict[str, Any], argument_index: int) -> None:
        self.value = value
        self.path = Path(value["path"])
        self.kind = value["kind"]
        self.argument_index = argument_index
        self.identity = tuple(value.get("identity", ()))
        self.binding = value
        self.path_descriptor: int | None = None
        self.resolved_snapshot: PinnedTargetFile | None = None
        if self.kind in {"directory", "regular_file"}:
            flags = (
                (os.O_RDWR if value.get("method") == "flock" else os.O_RDONLY)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            if self.kind == "directory":
                flags |= getattr(os, "O_DIRECTORY", 0)
            self.path_descriptor = os.open(self.path, flags)
            if PROFILER._identity(os.fstat(self.path_descriptor)) != self.identity:
                os.close(self.path_descriptor)
                self.path_descriptor = None
                raise CaptureError("target runtime path changed while opening")
        elif self.kind == "symlinked_file":
            resolved = Path(value["resolved_path"])
            self.resolved_snapshot = PinnedTargetFile.open(
                resolved,
                value["sha256"],
                argument_index,
                require_executable=False,
                binding=value,
            )
        try:
            self.verify()
        except Exception:
            self.close()
            raise

    @property
    def descriptor(self) -> int | None:
        if self.path_descriptor is not None:
            return self.path_descriptor
        return None if self.resolved_snapshot is None else self.resolved_snapshot.descriptor

    @property
    def fd_path(self) -> str | None:
        descriptor = self.descriptor
        return None if descriptor is None else f"/proc/self/fd/{descriptor}"

    def verify(self) -> None:
        try:
            if self.kind == "directory":
                metadata = self.path.lstat()
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or PROFILER._identity(metadata) != self.identity
                    or self.path_descriptor is None
                    or PROFILER._identity(os.fstat(self.path_descriptor)) != self.identity
                ):
                    raise CaptureError("target runtime directory identity changed")
            elif self.kind == "regular_file":
                metadata = self.path.lstat()
                observed = PROFILER._identity(metadata)
                opened = (
                    () if self.path_descriptor is None
                    else PROFILER._identity(os.fstat(self.path_descriptor))
                )
                matches = (
                    observed[:4] == self.identity[:4] and opened[:4] == self.identity[:4]
                    if self.value.get("method") == "flock"
                    else observed == self.identity and opened == self.identity
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or self.path_descriptor is None
                    or not matches
                ):
                    raise CaptureError("target runtime file identity changed")
            elif self.kind == "symlinked_file":
                if self.path.resolve(strict=True) != Path(self.value["resolved_path"]):
                    raise CaptureError("target symlinked runtime path target changed")
                assert self.resolved_snapshot is not None
                self.resolved_snapshot.verify()
            else:
                raise CaptureError("target runtime path kind differs")
        except OSError as error:
            raise CaptureError(f"target runtime path is unavailable: {error}") from error

    def close(self) -> None:
        if self.path_descriptor is not None:
            os.close(self.path_descriptor)
            self.path_descriptor = None
        if self.resolved_snapshot is not None:
            self.resolved_snapshot.close()


def named_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "nlink": metadata.st_nlink,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


class PinnedFdMap:
    def __init__(self, descriptor: int, value: dict[str, Any], data: bytes) -> None:
        self.descriptor = descriptor
        self.value = value
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()

    @classmethod
    def create(
        cls,
        target_value: dict[str, Any],
        snapshots: list[Any],
    ) -> "PinnedFdMap":
        by_path: dict[str, Any] = {}
        bindings: list[dict[str, Any]] = []
        for snapshot in snapshots[1:]:
            binding = getattr(snapshot, "binding", None)
            descriptor = getattr(snapshot, "descriptor", None)
            path = getattr(snapshot, "path", None)
            if not isinstance(binding, dict) or type(descriptor) is not int or not isinstance(path, Path):
                raise CaptureError("target FD map source contract differs")
            logical_path = str(binding["path"])
            if logical_path in by_path:
                raise CaptureError("target FD map logical path is duplicated")
            metadata = os.fstat(descriptor)
            expected_sha256 = getattr(snapshot, "sha256", None)
            if expected_sha256 is None and binding.get("kind") == "symlinked_file":
                expected_sha256 = binding.get("sha256")
            record = {
                "role": binding["role"],
                "logical_path": logical_path,
                "resolved_path": (
                    str(snapshot.resolved_snapshot.path)
                    if isinstance(snapshot, PinnedRuntimePath)
                    and snapshot.resolved_snapshot is not None
                    else None
                ),
                "descriptor": descriptor,
                "kind": binding["kind"] if "kind" in binding else "regular_file",
                "closure": binding["closure"],
                "method": binding["method"],
                "identity": named_identity(metadata),
                "sha256": expected_sha256,
            }
            if (
                binding["method"] in {"exec", "dlopen", "read"}
                and (
                    not isinstance(expected_sha256, str)
                    or HELPER_SHA256_RE.fullmatch(expected_sha256) is None
                )
            ):
                raise CaptureError("target FD map content SHA coverage differs")
            bindings.append(record)
            by_path[logical_path] = snapshot
        expected_count = (
            len(target_value["input_files"])
            + len(target_value["runtime_paths"])
            + len(target_value["control_files"])
        )
        if len(bindings) != expected_count:
            raise CaptureError("target FD map coverage differs")
        value: dict[str, Any] = {
            "schema_version": FD_MAP_SCHEMA,
            "status": "bound",
            "map_sha256": None,
            "logical_argv_sha256": hashlib.sha256(canonical(target_value["argv"])).hexdigest(),
            "closure_contract": target_value["closure_contract"],
            "bindings": sorted(bindings, key=lambda item: (item["role"], item["logical_path"])),
        }
        value["map_sha256"] = self_hash(value, "map_sha256")
        data = canonical(value) + b"\n"
        if len(data) > FD_MAP_MAX_BYTES:
            raise CaptureError("target FD map exceeds the byte bound")
        flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
        try:
            descriptor = os.memfd_create("ullm-aq4-p3-fd-map", flags)
        except (AttributeError, OSError) as error:
            raise CaptureError(f"sealed target FD map creation failed: {error}") from error
        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise CaptureError("target FD map write failed")
                offset += written
            os.lseek(descriptor, 0, os.SEEK_SET)
            required_seals = (
                getattr(fcntl, "F_SEAL_SEAL", 0)
                | getattr(fcntl, "F_SEAL_SHRINK", 0)
                | getattr(fcntl, "F_SEAL_GROW", 0)
                | getattr(fcntl, "F_SEAL_WRITE", 0)
            )
            if not required_seals:
                raise CaptureError("target FD map sealing is unavailable")
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
            result = cls(descriptor, value, data)
            result.verify()
            return result
        except Exception:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        required_seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        if fcntl.fcntl(self.descriptor, fcntl.F_GET_SEALS) & required_seals != required_seals:
            raise CaptureError("target FD map seals differ")
        if os.pread(self.descriptor, len(self.data) + 1, 0) != self.data:
            raise CaptureError("target FD map bytes changed")
        for item in self.value["bindings"]:
            observed = named_identity(os.fstat(item["descriptor"]))
            expected = item["identity"]
            stable_keys = {"device", "inode", "mode", "nlink"}
            matches = (
                all(observed[key] == expected[key] for key in stable_keys)
                if item["method"] == "flock"
                else all(
                    observed[key] == expected[key]
                    for key in {"device", "inode", "mode", "nlink", "size", "mtime_ns"}
                )
                if item["method"] != "pre_post_guard"
                else observed == expected
            )
            if not matches:
                raise CaptureError(f"target FD map identity changed: {item['role']}")

    def close(self) -> None:
        os.close(self.descriptor)


def process_group_alive(process_group: int, process: subprocess.Popen[Any]) -> bool:
    process.poll()
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(process: subprocess.Popen[Any]) -> bool:
    process_group = process.pid
    for value, wait_seconds in (
        (signal.SIGINT, 0.5),
        (signal.SIGTERM, 0.5),
        (signal.SIGKILL, 5.0),
    ):
        if not process_group_alive(process_group, process):
            break
        try:
            os.killpg(process_group, value)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait_seconds
        while (
            process_group_alive(process_group, process)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.kill()
        process.wait()
    return not process_group_alive(process_group, process)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value, allow_nan=False))
    clone[field] = None
    return hashlib.sha256(canonical(clone)).hexdigest()


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CaptureError(f"{label} fields differ")
    return value


def _validate_key_type_summary(keys: Any, types: Any, label: str) -> None:
    if (
        not isinstance(keys, list)
        or any(not isinstance(key, str) or not key for key in keys)
        or keys != sorted(keys)
        or len(keys) != len(set(keys))
        or not isinstance(types, dict)
        or set(types) != set(keys)
        or any(
            kind not in {"null", "boolean", "integer", "number", "string", "array", "object", "omitted"}
            for kind in types.values()
        )
    ):
        raise CaptureError(f"{label} key/type summary differs")


def _validate_safe_scalar(value: Any, label: str) -> None:
    item = _require_exact_keys(
        value,
        {"present", "json_type", "value", "string_length", "canonical_sha256"},
        label,
    )
    if type(item["present"]) is not bool:
        raise CaptureError(f"{label} presence differs")
    if item["json_type"] not in {"absent", "null", "boolean", "integer", "number", "string", "array", "object"}:
        raise CaptureError(f"{label} JSON type differs")
    if item["present"] != (item["json_type"] != "absent"):
        raise CaptureError(f"{label} presence/type differs")
    if item["canonical_sha256"] is not None and not _sha256_string(item["canonical_sha256"]):
        raise CaptureError(f"{label} SHA-256 differs")
    if item["present"] != (item["canonical_sha256"] is not None):
        raise CaptureError(f"{label} SHA-256 presence differs")
    if item["string_length"] is not None and (type(item["string_length"]) is not int or item["string_length"] < 0):
        raise CaptureError(f"{label} string length differs")
    if item["value"] is not None and not isinstance(item["value"], (bool, int, float, str)):
        raise CaptureError(f"{label} safe value differs")


def validate_ready_candidate_audit(value: Any) -> dict[str, Any]:
    audit = _require_exact_keys(
        value,
        {
            "schema_version",
            "audit_sha256",
            "raw",
            "top_level",
            "safe_scalars",
            "resident_session_id",
            "nested",
            "validation",
        },
        "ready candidate audit",
    )
    if audit["schema_version"] != READY_CANDIDATE_AUDIT_SCHEMA or not _sha256_string(audit["audit_sha256"]):
        raise CaptureError("ready candidate audit identity differs")
    if self_hash(audit, "audit_sha256") != audit["audit_sha256"]:
        raise CaptureError("ready candidate audit self-hash differs")
    if len(canonical(audit)) > MAX_READY_CANDIDATE_MARKER_BYTES:
        raise CaptureError("ready candidate audit exceeds bound")
    raw = _require_exact_keys(audit["raw"], {"byte_count", "raw_sha256"}, "ready candidate raw")
    if type(raw["byte_count"]) is not int or raw["byte_count"] < 0 or not _sha256_string(raw["raw_sha256"]):
        raise CaptureError("ready candidate raw binding differs")
    top = _require_exact_keys(audit["top_level"], {"key_count", "keys", "key_types"}, "ready candidate top level")
    if type(top["key_count"]) is not int or top["key_count"] < 0:
        raise CaptureError("ready candidate key count differs")
    _validate_key_type_summary(top["keys"], top["key_types"], "ready candidate top level")
    scalars = _require_exact_keys(audit["safe_scalars"], {"event", "schema_version", "model_loads"}, "ready candidate safe scalars")
    for name in ("event", "schema_version", "model_loads"):
        _validate_safe_scalar(scalars[name], f"ready candidate {name}")
    session = _require_exact_keys(
        audit["resident_session_id"],
        {"present", "json_type", "string_length", "canonical_sha256"},
        "ready candidate session ID",
    )
    if (
        type(session["present"]) is not bool
        or session["json_type"] not in {"absent", "null", "boolean", "integer", "number", "string", "array", "object"}
        or session["present"] != (session["json_type"] != "absent")
        or (session["string_length"] is not None and (type(session["string_length"]) is not int or session["string_length"] < 0))
        or (session["canonical_sha256"] is not None and not _sha256_string(session["canonical_sha256"]))
        or session["present"] != (session["canonical_sha256"] is not None)
    ):
        raise CaptureError("ready candidate session ID summary differs")
    nested = _require_exact_keys(audit["nested"], {"driver_identity", "served_model_binding"}, "ready candidate nested summaries")
    for name in ("driver_identity", "served_model_binding"):
        item = _require_exact_keys(
            nested[name],
            {"present", "json_type", "canonical_sha256", "keys", "key_types"},
            f"ready candidate {name}",
        )
        if (
            type(item["present"]) is not bool
            or item["json_type"] not in {"absent", "null", "boolean", "integer", "number", "string", "array", "object"}
            or item["present"] != (item["json_type"] != "absent")
            or (item["canonical_sha256"] is not None and not _sha256_string(item["canonical_sha256"]))
            or item["present"] != (item["canonical_sha256"] is not None)
        ):
            raise CaptureError(f"ready candidate {name} summary differs")
        _validate_key_type_summary(item["keys"], item["key_types"], f"ready candidate {name}")
    validation = _require_exact_keys(audit["validation"], {"status", "reason_code", "predicates"}, "ready candidate validation")
    predicate_keys = {
        "field_set_exact",
        "event_is_ready",
        "schema_version_exact",
        "model_loads_is_integer",
        "model_loads_is_one",
        "resident_session_id_is_string",
        "resident_session_id_nonempty",
    }
    if (
        validation["status"] not in {"passed", "failed"}
        or not isinstance(validation["reason_code"], str)
        or not validation["reason_code"]
        or not isinstance(validation["predicates"], dict)
        or set(validation["predicates"]) != predicate_keys
        or any(type(child) is not bool for child in validation["predicates"].values())
    ):
        raise CaptureError("ready candidate validation summary differs")
    return audit


def _ready_capture_binding(
    *,
    status: str,
    reason_code: str,
    stream_sha256: str | None,
    marker_count: int,
    marker_sha256: str | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": READY_CANDIDATE_CAPTURE_SCHEMA,
        "self_sha256": None,
        "status": status,
        "reason_code": reason_code,
        "source_stream": "rocprof.stderr",
        "source_stream_sha256": stream_sha256,
        "marker_count": marker_count,
        "marker_sha256": marker_sha256,
        "audit_sha256": None if audit is None else audit["audit_sha256"],
        "audit": audit,
    }
    value["self_sha256"] = self_hash(value, "self_sha256")
    return value


def parse_ready_candidate_marker(path: Path, stream_sha256: str | None) -> dict[str, Any]:
    markers: list[bytes] = []
    invalid_reason: str | None = None
    with path.open("rb", buffering=0) as source:
        while True:
            line = source.readline(MAX_READY_CANDIDATE_MARKER_BYTES + 2)
            if not line:
                break
            if len(line) > MAX_READY_CANDIDATE_MARKER_BYTES and not line.endswith(b"\n"):
                starts_marker = line.startswith(READY_CANDIDATE_MARKER_PREFIX)
                while line and not line.endswith(b"\n"):
                    line = source.readline(MAX_READY_CANDIDATE_MARKER_BYTES + 2)
                if starts_marker:
                    invalid_reason = "ready_candidate_marker_oversize"
                continue
            if line.startswith(READY_CANDIDATE_MARKER_PREFIX):
                if len(line.removesuffix(b"\n")) > MAX_READY_CANDIDATE_MARKER_BYTES:
                    invalid_reason = "ready_candidate_marker_oversize"
                    continue
                markers.append(line)
    if not markers:
        return _ready_capture_binding(
            status="invalid" if invalid_reason else "absent",
            reason_code=invalid_reason or "ready_candidate_marker_absent",
            stream_sha256=stream_sha256,
            marker_count=0,
        )
    if invalid_reason is not None or len(markers) != 1:
        return _ready_capture_binding(
            status="invalid",
            reason_code=invalid_reason or "ready_candidate_marker_count_differs",
            stream_sha256=stream_sha256,
            marker_count=len(markers),
        )
    marker = markers[0]
    marker_sha256 = hashlib.sha256(marker).hexdigest()
    if not marker.endswith(b"\n") or marker.endswith(b"\r\n"):
        return _ready_capture_binding(
            status="invalid",
            reason_code="ready_candidate_marker_termination_differs",
            stream_sha256=stream_sha256,
            marker_count=1,
            marker_sha256=marker_sha256,
        )
    payload = marker[len(READY_CANDIDATE_MARKER_PREFIX) : -1]
    try:
        def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, child in items:
                if key in value:
                    raise CaptureError(f"duplicate ready candidate audit key: {key}")
                value[key] = child
            return value

        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(CaptureError(f"non-finite ready candidate audit value: {item}")),
        )
        audit = validate_ready_candidate_audit(parsed)
        if audit["validation"]["status"] != "failed":
            raise CaptureError("ready candidate marker does not describe a failure")
    except (CaptureError, UnicodeError, json.JSONDecodeError, ValueError):
        return _ready_capture_binding(
            status="invalid",
            reason_code="ready_candidate_marker_payload_invalid",
            stream_sha256=stream_sha256,
            marker_count=1,
            marker_sha256=marker_sha256,
        )
    return _ready_capture_binding(
        status="valid",
        reason_code="ready_candidate_marker_bound",
        stream_sha256=stream_sha256,
        marker_count=1,
        marker_sha256=marker_sha256,
        audit=audit,
    )


def ref(snapshot: Any) -> dict[str, str]:
    return {"path": str(snapshot.path), "sha256": snapshot.sha256}


def pinned_profiler_version(profiler: PinnedProfiler) -> dict[str, Any]:
    profiler.verify()
    with tempfile.TemporaryFile() as output:
        profiler.verify()
        process = subprocess.Popen(
            [profiler.fd_path, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            pass_fds=(profiler.descriptor,),
            start_new_session=True,
        )
        try:
            try:
                return_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired as error:
                if not terminate_process_group(process):
                    raise CaptureError(
                        "profiler version query timed out and process group cleanup failed"
                    ) from error
                raise CaptureError("profiler version query timed out") from error
        finally:
            profiler.verify()
        size = output.tell()
        if return_code != 0 or size > 64 * 1024:
            raise CaptureError("profiler version query failed")
        output.seek(0)
        version_output = output.read(64 * 1024 + 1)
    text = version_output.decode("utf-8", errors="strict").strip()
    version = re.search(r"version:\s*([^\s]+)", text)
    rocm = re.search(r"rocm_version:\s*([^\s]+)", text)
    if version is None:
        raise CaptureError("profiler version output schema differs")
    return {
        **profiler.evidence(),
        "version": version.group(1),
        "rocm_version": rocm.group(1) if rocm else None,
        "version_output_sha256": hashlib.sha256(version_output).hexdigest(),
    }


def validate_target_manifest_root(value: dict[str, Any]) -> None:
    fields = {
        "schema_version", "status", "manifest_sha256", "argv", "environment",
        "input_files", "runtime_paths", "control_files", "output_paths",
        "closure_contract", "capture_helpers", "authorization",
    }
    PRODUCER.exact(value, fields, "target command manifest")
    if value.get("schema_version") != TARGET_SCHEMA or value.get("status") != "bound":
        raise CaptureError("target command manifest schema/status differs")
    declared = PRODUCER.digest(value.get("manifest_sha256"), "target command manifest hash")
    if declared != self_hash(value, "manifest_sha256"):
        raise CaptureError("target command manifest self-hash differs")
    environment = value.get("environment")
    if (
        not isinstance(environment, dict)
        or not environment
        or len(environment) > 128
        or any(
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or not isinstance(item, str)
            or "\x00" in item
            or len(key) > 256
            or len(item) > 16 * 1024
            for key, item in environment.items()
        )
    ):
        raise CaptureError("target command environment is invalid")
    authorization = value.get("authorization")
    if authorization != {
        "maximum_invocations": 1,
        "target_role": "profile_runner_only",
        "promotion_eligible": False,
    }:
        raise CaptureError("target command authorization differs")
    if value.get("capture_helpers") != capture_helper_contract():
        raise CaptureError("target command capture helper binding differs")
    if value.get("closure_contract") != {
        "code_execution_closure": "pinned_fd",
        "control_input_closure": "pinned_fd",
        "device_lock_closure": "pinned_fd",
        "data_integrity": "trusted_pre_post_guarded",
    }:
        raise CaptureError("target command closure contract differs")


def validate_fd_binding_fields(item: dict[str, Any], label: str) -> None:
    role = item.get("role")
    closure = item.get("closure")
    method = item.get("method")
    if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", role):
        raise CaptureError(f"{label} role differs")
    allowed = {
        "code_execution": {"exec", "dlopen"},
        "control_input": {"read"},
        "device_lock": {"flock"},
        "data_integrity": {"pre_post_guard"},
    }
    if closure not in allowed or method not in allowed[closure]:
        raise CaptureError(f"{label} closure/method differs")


def load_target_command_manifest(
    path: Path,
    expected_sha256: str,
    *,
    allow_existing_outputs: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    manifest_snapshot = PinnedTargetManifest.open(path, expected_sha256)
    try:
        data = manifest_snapshot.read_verified()
        parser_snapshot = PRODUCER.Snapshot(
            manifest_snapshot.path,
            PRODUCER.file_identity(os.fstat(manifest_snapshot.descriptor)),
            manifest_snapshot.sha256,
            data,
        )
        value = PRODUCER.parse_json(parser_snapshot, "target command manifest")
        validate_target_manifest_root(value)
        argv = value.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise CaptureError("target command argv is invalid")
        inputs = value.get("input_files")
        runtime_paths = value.get("runtime_paths")
        control_files = value.get("control_files")
        outputs = value.get("output_paths")
        if (
            not isinstance(inputs, list)
            or not inputs
            or not isinstance(runtime_paths, list)
            or not isinstance(control_files, list)
            or not isinstance(outputs, list)
        ):
            raise CaptureError("target command path bindings are invalid")
        snapshots: list[Any] = [manifest_snapshot]
        classified: set[int] = set()
        for number, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise CaptureError("target input binding must be an object")
            PRODUCER.exact(
                item,
                {"argument_index", "path", "sha256", "executable", "role", "closure", "method"},
                f"target input {number}",
            )
            validate_fd_binding_fields(item, f"target input {number}")
            index = PRODUCER.count(
                item.get("argument_index"), f"target input {number} index"
            )
            if index >= len(argv) or index in classified or argv[index] != item.get("path"):
                raise CaptureError("target input index/path binding differs")
            if type(item.get("executable")) is not bool:
                raise CaptureError("target input executable flag must be boolean")
            snapshot = PinnedTargetFile.open(
                Path(item["path"]),
                PRODUCER.digest(item.get("sha256"), f"target input {number} hash"),
                index,
                require_executable=item["executable"],
                binding=item,
            )
            snapshots.append(snapshot)
            classified.add(index)
        if 0 not in classified or not any(
            item.get("argument_index") == 0 and item.get("executable") is True
            for item in inputs
        ):
            raise CaptureError("target argv[0] executable is not hash-bound")
        for number, item in enumerate(runtime_paths):
            if not isinstance(item, dict):
                raise CaptureError("target runtime path binding must be an object")
            common = {"argument_index", "path", "kind", "role", "closure", "method"}
            kind = item.get("kind")
            expected = (
                common | {"identity"}
                if kind in {"directory", "regular_file"}
                else common | {"resolved_path", "sha256"}
                if kind == "symlinked_file"
                else set()
            )
            if not expected:
                raise CaptureError("target runtime path kind differs")
            PRODUCER.exact(item, expected, f"target runtime path {number}")
            validate_fd_binding_fields(item, f"target runtime path {number}")
            index = PRODUCER.count(
                item.get("argument_index"), f"target runtime path {number} index"
            )
            if (
                index >= len(argv)
                or index in classified
                or argv[index] != item.get("path")
                or not Path(item["path"]).is_absolute()
            ):
                raise CaptureError("target runtime path index/path binding differs")
            if kind in {"directory", "regular_file"}:
                identity = item.get("identity")
                if (
                    not isinstance(identity, list)
                    or len(identity) != 7
                    or any(type(part) is not int for part in identity)
                ):
                    raise CaptureError("target runtime path identity differs")
            else:
                PRODUCER.digest(item.get("sha256"), "target symlinked runtime path hash")
                resolved = item.get("resolved_path")
                if not isinstance(resolved, str) or not Path(resolved).is_absolute():
                    raise CaptureError("target symlinked runtime path resolution differs")
            snapshots.append(PinnedRuntimePath(item, index))
            classified.add(index)
        indirect_paths: set[str] = set()
        for number, item in enumerate(control_files):
            if not isinstance(item, dict):
                raise CaptureError("target control binding must be an object")
            PRODUCER.exact(
                item,
                {"path", "sha256", "role", "closure", "method"},
                f"target control {number}",
            )
            validate_fd_binding_fields(item, f"target control {number}")
            path_text = item.get("path")
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
                or path_text in indirect_paths
                or any(binding.get("path") == path_text for binding in inputs)
                or any(binding.get("path") == path_text for binding in runtime_paths)
            ):
                raise CaptureError("target control path coverage differs")
            snapshot = PinnedTargetFile.open(
                Path(path_text),
                PRODUCER.digest(item.get("sha256"), f"target control {number} hash"),
                None,
                require_executable=False,
                binding=item,
            )
            snapshots.append(snapshot)
            indirect_paths.add(path_text)
        for number, item in enumerate(outputs):
            if not isinstance(item, dict):
                raise CaptureError("target output binding must be an object")
            PRODUCER.exact(
                item, {"argument_index", "path"}, f"target output {number}"
            )
            index = PRODUCER.count(
                item.get("argument_index"), f"target output {number} index"
            )
            target = item.get("path")
            if (
                index >= len(argv)
                or index in classified
                or argv[index] != target
                or not isinstance(target, str)
                or not Path(target).is_absolute()
            ):
                raise CaptureError("target output index/path binding differs")
            if Path(target).is_symlink():
                raise CaptureError("target output path must not be a symlink")
            if not allow_existing_outputs and Path(target).exists():
                raise CaptureError("target output path already exists")
            PROFILER.canonical_path(
                Path(target).parent, f"target output {number} parent"
            )
            classified.add(index)
        absolute_indices = {
            index for index, item in enumerate(argv) if Path(item).is_absolute()
        }
        if classified != absolute_indices:
            raise CaptureError("target absolute argv path coverage differs")
        return value, snapshots
    except Exception:
        for snapshot in reversed(locals().get("snapshots", [manifest_snapshot])):
            close = getattr(snapshot, "close", None)
            if callable(close):
                close()
        raise


def pinned_target_argv(value: dict[str, Any], snapshots: list[Any]) -> tuple[list[str], tuple[int, ...]]:
    argv = list(value["argv"])
    replacements: dict[int, str] = {}
    descriptors: list[int] = []
    for snapshot in snapshots[1:]:
        descriptor = getattr(snapshot, "descriptor", None)
        fd_path = getattr(snapshot, "fd_path", None)
        index = getattr(snapshot, "argument_index", None)
        binding = getattr(snapshot, "binding", {})
        if descriptor is None and fd_path is None:
            continue
        if type(descriptor) is not int or not isinstance(fd_path, str):
            raise CaptureError("target FD binding contract differs")
        descriptors.append(descriptor)
        if binding.get("role") not in {"python_interpreter", "resident_runner"}:
            continue
        if type(index) is not int or index in replacements or index >= len(argv):
            raise CaptureError("target FD binding coverage differs")
        replacements[index] = fd_path
    if set(replacements) != {0, 1}:
        raise CaptureError("target FD replacement coverage differs")
    for index, fd_path in replacements.items():
        argv[index] = fd_path
    if len(descriptors) != len(set(descriptors)):
        raise CaptureError("target FD descriptors are not unique")
    return argv, tuple(sorted(descriptors))


def close_target_snapshots(snapshots: list[Any]) -> None:
    for snapshot in reversed(snapshots):
        close = getattr(snapshot, "close", None)
        if callable(close):
            close()


def profiler_command(
    profiler: Any, output_directory: Path, output_name: str, runner_command: list[str]
) -> list[str]:
    if not output_directory.is_absolute():
        raise CaptureError("profile output directory must be absolute")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
        raise CaptureError("profile output name is unsafe")
    if not runner_command or any(not isinstance(item, str) or not item for item in runner_command):
        raise CaptureError("runner command is empty or invalid")
    if not Path(runner_command[0]).is_absolute():
        raise CaptureError("runner executable path must be absolute")
    executable = profiler.fd_path if isinstance(profiler, PinnedProfiler) else str(profiler.path)
    return [
        executable,
        "--log-level",
        "error",
        "--kernel-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--marker-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(output_directory),
        "--output-file",
        output_name,
        "--",
        *runner_command,
    ]


def _run_profile(
    command: list[str],
    output_directory: Path,
    timeout: float,
    *,
    pass_fds: tuple[int, ...] = (),
    spawn_verifier: Any = None,
    environment: dict[str, str] | None = None,
    on_rocprof_started: Any = None,
) -> None:
    if timeout <= 0.0:
        raise CaptureError("profile timeout must be positive")
    if not output_directory.is_absolute():
        raise CaptureError("profile output directory must be absolute")
    PROFILER.canonical_path(output_directory.parent, "profile output parent")
    if output_directory.exists() or output_directory.is_symlink():
        raise CaptureError("profile output directory already exists")
    output_directory.mkdir(mode=0o700)
    stdout_path = output_directory / "rocprof.stdout"
    stderr_path = output_directory / "rocprof.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        if spawn_verifier is not None:
            spawn_verifier()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
            pass_fds=pass_fds,
            env=None if environment is None else dict(environment),
        )
        if on_rocprof_started is not None:
            try:
                on_rocprof_started()
            except Exception:
                if not terminate_process_group(process):
                    raise CaptureError("rocprof start callback failed and process group cleanup failed")
                raise
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            if not terminate_process_group(process):
                raise CaptureError(
                    "rocprof diagnostic capture timed out and process group cleanup failed"
                ) from error
            raise CaptureError("rocprof diagnostic capture timed out") from error
    if process_group_alive(process.pid, process):
        if not terminate_process_group(process):
            raise CaptureError("rocprof exited with descendant processes and process group cleanup failed")
        raise CaptureError("rocprof exited with descendant processes; descendants were terminated")
    if return_code != 0:
        suffix = " (possible OOM/SIGKILL)" if return_code in {-9, 9, 137} else ""
        raise CaptureError(f"rocprof diagnostic capture failed with exit {return_code}{suffix}")


def write_failure_evidence(
    output_directory: Path,
    reason: str,
    command: list[str],
    context: dict[str, Any] | None,
    *,
    effective_command: list[str] | None = None,
) -> None:
    path = output_directory / "capture-failure.json"
    if path.exists() or path.is_symlink():
        return
    streams: dict[str, Any] = {}
    for name in ("rocprof.stdout", "rocprof.stderr"):
        stream = output_directory / name
        if stream.is_file() and not stream.is_symlink():
            snapshot = PROFILER.capture(stream.resolve(), f"failure stream {name}")
            streams[name] = {
                "bytes": stream.stat().st_size,
                "sha256": snapshot.sha256,
            }
    stderr_path = output_directory / "rocprof.stderr"
    if "rocprof.stderr" in streams:
        try:
            ready_candidate_audit = parse_ready_candidate_marker(
                stderr_path,
                streams["rocprof.stderr"]["sha256"],
            )
        except OSError:
            ready_candidate_audit = _ready_capture_binding(
                status="invalid",
                reason_code="ready_candidate_source_stream_unavailable",
                stream_sha256=streams["rocprof.stderr"]["sha256"],
                marker_count=0,
            )
    else:
        ready_candidate_audit = _ready_capture_binding(
            status="absent",
            reason_code="ready_candidate_source_stream_absent",
            stream_sha256=None,
            marker_count=0,
        )
    cleanup_complete = "cleanup failed" not in reason
    value = {
        "schema_version": FAILURE_SCHEMA,
        "status": "failed",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "failure_sha256": None,
        "reason": reason,
        "rocprof_child_new_session": True,
        "outer_harness_signalled": False,
        "process_group_cleanup_complete": cleanup_complete,
        "children_state_known": cleanup_complete,
        "children_remaining": [],
        "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
        "effective_command_sha256": hashlib.sha256(canonical(effective_command if effective_command is not None else command)).hexdigest(),
        "context": context or {},
        "streams": streams,
        "ready_candidate_audit": ready_candidate_audit,
    }
    value["failure_sha256"] = self_hash(value, "failure_sha256")
    write_json_atomic(path, value)
    path.chmod(0o444)


def run_profile(
    command: list[str],
    output_directory: Path,
    timeout: float,
    *,
    pass_fds: tuple[int, ...] = (),
    verifier: Any = None,
    failure_context: dict[str, Any] | None = None,
    environment: dict[str, str] | None = None,
    on_rocprof_started: Any = None,
    logical_command: list[str] | None = None,
) -> None:
    if logical_command is not None and (not logical_command or any(not isinstance(item, str) or not item for item in logical_command)):
        raise CaptureError("logical profile command is invalid")
    if verifier is not None:
        verifier()
    output_preexisted = output_directory.exists() or output_directory.is_symlink()
    failure: BaseException | None = None
    reason: str | None = None
    try:
        _run_profile(
            command,
            output_directory,
            timeout,
            pass_fds=pass_fds,
            spawn_verifier=verifier,
            environment=environment,
            on_rocprof_started=on_rocprof_started,
        )
    except (
        CaptureError,
        PRODUCER.ProducerError,
        PROFILER.ProfileError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        failure = error
        reason = str(error)
    try:
        if verifier is not None:
            verifier()
    except (
        CaptureError,
        PRODUCER.ProducerError,
        PROFILER.ProfileError,
        OSError,
    ) as error:
        if failure is None:
            failure = error
            reason = f"post-spawn input verification failed: {error}"
        else:
            reason = f"{reason}; post-spawn input verification failed: {error}"
    if failure is not None:
        assert reason is not None
        if (
            not output_preexisted
            and output_directory.is_dir()
            and not output_directory.is_symlink()
        ):
                write_failure_evidence(
                    output_directory,
                    reason,
                    logical_command if logical_command is not None else command,
                    failure_context,
                    effective_command=command,
                )
        if isinstance(failure, CaptureError) and reason == str(failure):
            raise failure
        raise CaptureError(reason) from failure


def discover(output_directory: Path) -> dict[str, Path]:
    patterns = {
        "kernel": "*_kernel_trace.csv",
        "hip_api": "*_hip_api_trace.csv",
        "memory_copy": "*_memory_copy_trace.csv",
        "marker": "*_marker_api_trace.csv",
    }
    result: dict[str, Path] = {}
    for kind, pattern in patterns.items():
        matches = sorted(output_directory.rglob(pattern))
        if len(matches) != 1:
            raise CaptureError(f"expected exactly one {kind} trace, got {len(matches)}")
        result[kind] = matches[0]
    return result


def csv_rows(snapshot: Any, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeError as error:
        raise CaptureError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise CaptureError(f"{label} header is missing or duplicated")
    rows: list[dict[str, str]] = []
    for line, row in enumerate(reader, 2):
        if len(rows) >= MAX_ROWS or None in row or any(value is None for value in row.values()):
            raise CaptureError(f"{label} row {line} is invalid")
        rows.append({key: value for key, value in row.items()})
    return fields, rows


def one_column(fields: list[str], aliases: tuple[str, ...], label: str) -> str:
    matches = [field for field in aliases if field in fields]
    if len(matches) != 1:
        raise CaptureError(f"trace must have exactly one {label} column")
    return matches[0]


def interval_columns(fields: list[str]) -> tuple[str, str]:
    return (
        one_column(fields, ("Start_Timestamp", "BeginNs", "start_ns"), "start"),
        one_column(fields, ("End_Timestamp", "EndNs", "end_ns"), "end"),
    )


def parse_marker_name(name: str) -> dict[str, str]:
    parts = name.split("/")
    if not parts or parts[0] != MARKER_PREFIX:
        raise CaptureError(f"unknown marker name: {name}")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise CaptureError(f"invalid marker field: {part}")
        key, value = part.split("=", 1)
        if key in values or not value:
            raise CaptureError(f"duplicate or empty marker field: {key}")
        values[key] = value
    if set(values) != MARKER_KEYS:
        raise CaptureError("marker fields differ")
    return values


def markers(snapshot: Any, raw: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    fields, rows = csv_rows(snapshot, "marker trace")
    name_column = one_column(fields, ("Name", "Marker_Name", "name"), "marker name")
    start_column, end_column = interval_columns(fields)
    expected_session = raw["resident"]["session_id"]
    result: list[dict[str, Any]] = []
    previous_end = -1
    for row in rows:
        values = parse_marker_name(row[name_column].strip())
        try:
            index = int(values["run_index"])
            start = int(row[start_column])
            end = int(row[end_column])
        except ValueError as error:
            raise CaptureError("marker integer field is invalid") from error
        expected_kind = "warmup" if index < 2 else "measured"
        if (
            index != len(result)
            or index > 11
            or values["run_kind"] != expected_kind
            or values["run_id"] != run_id
            or values["session_id"] != expected_session
            or values["case_id"] != raw["case_id"]
            or values["case_sha256"] != raw["case_sha256"]
            or start < 0
            or end <= start
            or start < previous_end
        ):
            raise CaptureError("marker order/kind/identity/interval differs")
        result.append({**values, "run_index": index, "start_ns": start, "end_ns": end})
        previous_end = end
    if len(result) != 12:
        raise CaptureError("marker trace must contain exactly 12 balanced run ranges")
    return result


def rows_by_marker(
    fields: list[str], rows: list[dict[str, str]], ranges: list[dict[str, Any]], label: str
) -> dict[int, list[dict[str, str]]]:
    start_column, end_column = interval_columns(fields)
    result = {index: [] for index in range(12)}
    for line, row in enumerate(rows, 2):
        try:
            start, end = int(row[start_column]), int(row[end_column])
        except ValueError as error:
            raise CaptureError(f"{label} row {line} clock is invalid") from error
        if start < 0 or end <= start:
            raise CaptureError(f"{label} row {line} interval is invalid")
        containing = [item for item in ranges if item["start_ns"] <= start and end <= item["end_ns"]]
        crossing = [item for item in ranges if start < item["end_ns"] and end > item["start_ns"]]
        if len(containing) > 1 or (crossing and len(containing) != 1):
            raise CaptureError(f"{label} row {line} crosses a run marker")
        if containing:
            result[containing[0]["run_index"]].append(row)
    return result


def validate_memory_copy_rows(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Name", "Kind", "Direction", "Operation", "name"), "memory copy kind"
    )
    seen: set[str] = set()
    correlation_column = one_column(
        fields, ("Correlation_Id", "Correlation_ID", "Index", "correlation_id"),
        "memory correlation id",
    )
    for line, row in enumerate(rows, 2):
        correlation = row[correlation_column].strip()
        kind = re.sub(r"[^a-z0-9]", "", row[name_column].lower())
        if not correlation or correlation in seen:
            raise CaptureError(f"memory copy row {line} correlation differs")
        seen.add(correlation)
        if kind not in MEMORY_COPY_KINDS:
            raise CaptureError(f"unknown memory copy operation: {row[name_column]}")


def validate_all_kernel_names(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Kernel_Name", "KernelName", "Name", "kernel_name"), "kernel name"
    )
    for row in rows:
        name = row[name_column].strip()
        try:
            family = PROFILER.classify_kernel(name)
        except PROFILER.ProfileError as error:
            raise CaptureError(f"kernel family classification failed: {error}") from error
        if family is None:
            raise CaptureError(f"unknown kernel family in source trace: {name}")


def validate_all_hip_api_names(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Function", "Api_Name", "API_Name", "Name", "function"), "HIP API name"
    )
    known = (
        PRODUCER.D2H_APIS
        | PRODUCER.SYNC_APIS
        | PRODUCER.KNOWN_OTHER_MEMCPY_APIS
        | PRODUCER.KNOWN_OTHER_SYNC_APIS
    )
    for row in rows:
        raw_name = row[name_column].strip()
        name = PRODUCER.normalized_api_name(raw_name)
        if name in known:
            continue
        if "memcpy" in name or "synchron" in name:
            raise CaptureError(f"unknown transfer/synchronization HIP API: {raw_name}")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Any:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to overwrite split trace: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return PRODUCER.capture(path.resolve(), "split trace")


def capability(profiler_value: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": PRODUCER.CAPABILITY_SCHEMA,
        "status": "complete",
        "measurement_eligible": False,
        "capability_sha256": None,
        "tool": {"name": "rocprofv3", "version": profiler_value["version"]},
        "domains": {
            "kernel_dispatch": True,
            "hip_api": True,
            "memory_copy": True,
            "d2h_memcpy": True,
            "stream_synchronize": True,
            "device_synchronize": True,
        },
        "rocprof_config": {
            "kernel_trace": True,
            "hip_api_trace": True,
            "memory_copy_trace": True,
            "marker_trace": True,
            "api_filter": "all_functions",
        },
    }
    value["capability_sha256"] = PRODUCER.self_hash(value, "capability_sha256")
    return value


def validate_resident_evidence(
    identity_path: Path, summary_path: Path, raw_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any], str]:
    snapshots: list[Any] = []
    identity_snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    snapshots.append(identity_snapshot)
    identity_value = PRODUCER.parse_json(identity_snapshot, "identity")
    identity = PRODUCER.validate_identity(identity_value, identity_snapshot)
    summary_snapshot = PRODUCER.capture(summary_path.resolve(), "resident summary")
    snapshots.append(summary_snapshot)
    summary_value = PRODUCER.parse_json(summary_snapshot, "resident summary")
    run_id = PRODUCER.validate_summary(
        summary_value, summary_snapshot, identity, "diagnostic"
    )
    raw_snapshot = PRODUCER.capture(raw_path.resolve(), "resident raw")
    snapshots.append(raw_snapshot)
    raw_value = PRODUCER.parse_json(raw_snapshot, "resident raw")
    raw_run_id, _runs = PRODUCER.validate_raw(
        raw_value, identity, {run_id: summary_snapshot}, "diagnostic"
    )
    if raw_run_id != run_id:
        raise CaptureError("resident raw/summary run_id differs")
    return identity, summary_value, raw_value, snapshots, run_id


def _assemble(
    *,
    traces: dict[str, Path],
    identity_path: Path,
    summary_path: Path,
    raw_path: Path,
    profiler_value: dict[str, Any],
    command: list[str],
    output_directory: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    if artifact_path.exists() or artifact_path.is_symlink():
        raise CaptureError("capture artifact already exists")
    split_directory = output_directory / "measured-runs"
    if split_directory.exists() or split_directory.is_symlink():
        raise CaptureError("measured split directory already exists")
    identity, _summary, raw, evidence_snapshots, run_id = validate_resident_evidence(
        identity_path, summary_path, raw_path
    )
    trace_snapshots = {
        kind: PRODUCER.capture(path.resolve(), f"{kind} trace") for kind, path in traces.items()
    }
    if len({snapshot.sha256 for snapshot in trace_snapshots.values()}) != len(trace_snapshots):
        raise CaptureError("source trace bytes were reused across domains")
    ranges = markers(trace_snapshots["marker"], raw, run_id)
    parsed: dict[str, tuple[list[str], dict[int, list[dict[str, str]]]]] = {}
    for kind in ("kernel", "hip_api", "memory_copy"):
        fields, rows = csv_rows(trace_snapshots[kind], f"{kind} trace")
        if kind == "kernel":
            validate_all_kernel_names(fields, rows)
        elif kind == "hip_api":
            validate_all_hip_api_names(fields, rows)
        else:
            validate_memory_copy_rows(fields, rows)
        parsed[kind] = (fields, rows_by_marker(fields, rows, ranges, f"{kind} trace"))
    split_directory.mkdir(mode=0o700)
    capability_value = capability(profiler_value)
    capability_path = output_directory / "capture-capabilities.json"
    if capability_path.exists() or capability_path.is_symlink():
        raise CaptureError("capture capability output already exists")
    write_json_atomic(capability_path, capability_value)
    capability_snapshot = PRODUCER.capture(capability_path.resolve(), "capture capabilities")
    PRODUCER.validate_capture_capabilities(capability_value, "diagnostic")
    profile_runs: list[dict[str, Any]] = []
    split_snapshots: list[Any] = []
    used_kernel_traces: set[str] = set()
    used_api_traces: set[str] = set()
    for index in range(2, 12):
        kernel_fields, kernel_runs = parsed["kernel"]
        if not kernel_runs[index]:
            raise CaptureError(f"measured run {index} kernel trace is empty")
        kernel_output_fields = list(kernel_fields)
        kernel_rows = [dict(row) for row in kernel_runs[index]]
        if "Phase" not in kernel_output_fields:
            kernel_output_fields.append("Phase")
            for row in kernel_rows:
                row["Phase"] = "prefill"
        api_fields, api_runs = parsed["hip_api"]
        if not api_runs[index]:
            raise CaptureError(f"measured run {index} HIP API trace is empty")
        memory_fields, memory_runs = parsed["memory_copy"]
        kernel_snapshot = write_csv(
            split_directory / f"run-{index:02d}_kernel_trace.csv",
            kernel_output_fields,
            kernel_rows,
        )
        api_snapshot = write_csv(
            split_directory / f"run-{index:02d}_hip_api_trace.csv",
            api_fields,
            api_runs[index],
        )
        memory_snapshot = write_csv(
            split_directory / f"run-{index:02d}_memory_copy_trace.csv",
            memory_fields,
            memory_runs[index],
        )
        split_snapshots.extend((kernel_snapshot, api_snapshot, memory_snapshot))
        if kernel_snapshot.sha256 in used_kernel_traces or api_snapshot.sha256 in used_api_traces:
            raise CaptureError("measured kernel or HIP API trace bytes were reused")
        used_kernel_traces.add(kernel_snapshot.sha256)
        used_api_traces.add(api_snapshot.sha256)
        PRODUCER.parse_kernel_trace(kernel_snapshot, "paged-kv-table-validation-v1")
        PRODUCER.parse_hip_api_trace(api_snapshot, capability_value)
        profile_runs.append(
            {
                "schema_version": PRODUCER.PROFILE_BINDING_SCHEMA,
                "case_id": raw["case_id"],
                "case_sha256": raw["case_sha256"],
                "identity_sha256": identity["identity_sha256"],
                "resident_run_index": index,
                "measurement_eligible": False,
                "clock_domain": MARKER_CLOCK,
                "kernel_trace_complete": True,
                "hip_api_trace_complete": True,
                "capture_capabilities": ref(capability_snapshot),
                "kernel_trace": ref(kernel_snapshot),
                "hip_api_trace": ref(api_snapshot),
            }
        )
    artifact = {
        "schema_version": SCHEMA,
        "status": "complete_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "artifact_sha256": None,
        "binding": {
            "run_id": run_id,
            "resident_session_id": raw["resident"]["session_id"],
            "case_id": raw["case_id"],
            "case_sha256": raw["case_sha256"],
            "identity_sha256": identity["identity_sha256"],
            "device": identity["_resident_driver_identity"]["runtime_device"],
            "identity": ref(evidence_snapshots[0]),
            "resident_summary": ref(evidence_snapshots[1]),
            "resident_raw": ref(evidence_snapshots[2]),
        },
        "profiler": {
            **profiler_value,
            "command": command,
            "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
            "subprocess_profile_runs": 1,
        },
        "source_traces": {kind: ref(snapshot) for kind, snapshot in trace_snapshots.items()},
        "capture_capabilities": ref(capability_snapshot),
        "marker_contract": {
            "schema_version": MARKER_PREFIX,
            "clock_domain": MARKER_CLOCK,
            "range_count": 12,
            "warmup_indices": [0, 1],
            "measured_indices": list(range(2, 12)),
            "warmup_excluded": True,
        },
        "producer_profile_runs": profile_runs,
        "memory_copy_traces": [
            ref(snapshot) for snapshot in split_snapshots if "memory_copy" in snapshot.path.name
        ],
        "eligibility_blockers": [
            "rocprof instrumentation overhead forbids performance promotion",
            "one-case diagnostic evidence does not satisfy seven-prompt promotion coverage",
        ],
    }
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    for snapshot in [*evidence_snapshots, *trace_snapshots.values(), *split_snapshots, capability_snapshot]:
        snapshot.verify()
    write_json_atomic(artifact_path, artifact)
    artifact_path.chmod(0o444)
    return artifact


def assemble(**kwargs: Any) -> dict[str, Any]:
    output_directory = kwargs["output_directory"]
    artifact_path = kwargs["artifact_path"]
    failure_path = output_directory / "capture-failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        raise CaptureError("failure evidence exists; refusing to publish a success artifact")
    split_directory = output_directory / "measured-runs"
    capability_path = output_directory / "capture-capabilities.json"
    split_existed = split_directory.exists() or split_directory.is_symlink()
    capability_existed = capability_path.exists() or capability_path.is_symlink()
    artifact_existed = artifact_path.exists() or artifact_path.is_symlink()
    try:
        return _assemble(**kwargs)
    except Exception:
        if not split_existed and split_directory.exists() and not split_directory.is_symlink():
            shutil.rmtree(split_directory)
        if not capability_existed and capability_path.exists() and not capability_path.is_symlink():
            capability_path.unlink()
        if not artifact_existed and artifact_path.exists() and not artifact_path.is_symlink():
            artifact_path.unlink()
        raise


def main(
    argv: list[str] | None = None,
    *,
    on_rocprof_started: Any = None,
    on_runner_completed: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "assemble"))
    parser.add_argument("--profiler-path", type=Path, required=True)
    parser.add_argument("--profiler-sha256", required=True)
    parser.add_argument("--target-command-manifest", type=Path, required=True)
    parser.add_argument("--target-command-manifest-sha256", required=True)
    parser.add_argument("--profile-output-directory", type=Path, required=True)
    parser.add_argument("--profile-output-name", default="aq4-p3-diagnostic")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--resident-summary", type=Path, required=True)
    parser.add_argument("--resident-raw", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if (on_rocprof_started is not None or on_runner_completed is not None) and args.command != "capture":
        raise CaptureError("capture lifecycle callbacks are reserved for capture")
    pinned_profiler: PinnedProfiler | None = None
    pinned_target_manifest: PinnedTargetManifest | None = None
    pinned_fd_map: PinnedFdMap | None = None
    capture_completed = False
    failure_context: dict[str, Any] | None = None
    logical_command: list[str] | None = None
    target_snapshots: list[Any] = []
    try:
        if args.command == "capture" and (
            args.artifact.exists() or args.artifact.is_symlink()
        ):
            raise CaptureError("success artifact path already exists")
        pinned_profiler = PinnedProfiler.open(
            args.profiler_path, args.profiler_sha256
        )
        profiler_value = pinned_profiler_version(pinned_profiler)
        target_value, target_snapshots = load_target_command_manifest(
            args.target_command_manifest,
            args.target_command_manifest_sha256,
            allow_existing_outputs=args.command == "assemble",
        )
        pinned_target_manifest = target_snapshots[0]
        target_command, target_descriptors = pinned_target_argv(target_value, target_snapshots)
        pinned_fd_map = PinnedFdMap.create(target_value, target_snapshots)
        if FD_MAP_ENV in target_value["environment"]:
            raise CaptureError("target base environment reserves the FD map key")
        effective_environment = dict(target_value["environment"])
        effective_environment[FD_MAP_ENV] = str(pinned_fd_map.descriptor)
        profiler_value["target_command_manifest"] = ref(target_snapshots[0])
        profiler_value["target_environment"] = {
            "sha256": hashlib.sha256(canonical(target_value["environment"])).hexdigest(),
            "keys": sorted(target_value["environment"]),
            "exact_base_environment": True,
            "secret_material_recorded": False,
            "injected_fd_map_key": FD_MAP_ENV,
        }
        profiler_value["execution_closure"] = {
            **target_value["closure_contract"],
            "fd_map_schema": FD_MAP_SCHEMA,
            "fd_map_sha256": pinned_fd_map.value["map_sha256"],
            "fd_map_file_sha256": pinned_fd_map.sha256,
            "bindings": [
                {
                    "role": item["role"],
                    "logical_path": item["logical_path"],
                    "resolved_path": item["resolved_path"],
                    "kind": item["kind"],
                    "closure": item["closure"],
                    "method": item["method"],
                    "identity": item["identity"],
                    "sha256": item["sha256"],
                }
                for item in pinned_fd_map.value["bindings"]
            ],
            "capture_helpers": [
                {**item, "closure": "code_execution", "method": "verified_in_process"}
                for item in capture_helper_contract()
            ],
        }
        profiler_value["capture_helpers"] = capture_helper_contract()
        command = profiler_command(
            pinned_profiler,
            args.profile_output_directory,
            args.profile_output_name,
            target_command,
        )
        logical_command = [
            str(pinned_profiler.invocation),
            *profiler_command(
                pinned_profiler,
                args.profile_output_directory,
                args.profile_output_name,
                target_value["argv"],
            )[1:],
        ]
        failure_context = {
            "profiler": pinned_profiler.evidence(),
            "target_command_manifest": ref(target_snapshots[0]),
        }

        def verify_launch_inputs() -> None:
            pinned_profiler.verify()
            verify_capture_helpers()
            assert pinned_fd_map is not None
            pinned_fd_map.verify()
            try:
                for snapshot in target_snapshots:
                    snapshot.verify()
            except (PRODUCER.ProducerError, PROFILER.ProfileError) as error:
                raise CaptureError(f"target command binding changed: {error}") from error

        if args.command == "capture":
            run_profile(
                command,
                args.profile_output_directory,
                args.timeout,
                pass_fds=(pinned_profiler.descriptor, *target_descriptors, pinned_fd_map.descriptor),
                verifier=verify_launch_inputs,
                failure_context=failure_context,
                environment=effective_environment,
                on_rocprof_started=on_rocprof_started,
                logical_command=logical_command,
            )
            if on_runner_completed is not None:
                on_runner_completed()
            capture_completed = True
        elif not args.profile_output_directory.is_dir():
            raise CaptureError("assemble profile output directory is missing")
        verify_launch_inputs()
        for item in target_value["output_paths"]:
            target_output = Path(item["path"])
            if args.command == "capture" and (
                not target_output.exists() or target_output.is_symlink()
            ):
                raise CaptureError(
                    "target command did not create a non-symlink bound output path"
                )
        traces = discover(args.profile_output_directory)
        artifact = assemble(
            traces=traces,
            identity_path=args.identity,
            summary_path=args.resident_summary,
            raw_path=args.resident_raw,
            profiler_value=profiler_value,
            command=logical_command,
            output_directory=args.profile_output_directory,
            artifact_path=args.artifact,
        )
        capture_completed = False
        print(json.dumps({"status": artifact["status"], "promotion_eligible": False}, sort_keys=True))
        return 0
    except (
        CaptureError,
        PRODUCER.ProducerError,
        PROFILER.ProfileError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        if (
            args.command == "capture"
            and capture_completed
            and logical_command is not None
            and args.profile_output_directory.is_dir()
            and not args.profile_output_directory.is_symlink()
        ):
            write_failure_evidence(
                args.profile_output_directory,
                str(error),
                logical_command,
                failure_context,
                effective_command=command,
            )
        print(f"AQ4 P3 diagnostic rocprof capture failed: {error}", file=sys.stderr)
        return 1
    finally:
        if target_snapshots:
            close_target_snapshots(target_snapshots)
        if pinned_fd_map is not None:
            pinned_fd_map.close()
        if pinned_profiler is not None:
            pinned_profiler.close()


if __name__ == "__main__":
    raise SystemExit(main())
