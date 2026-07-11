#!/usr/bin/env python3
"""Fail-closed orchestration boundary for one full SQ8 OpenWebUI campaign."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import secrets
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, NoReturn, Protocol, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sq8_full_campaign_bundle import (  # noqa: E402
    AtomicCampaignDirectory,
    FileEvidence,
    PREVALIDATION_ROOT_FILES,
)
from sq8_openwebui_campaign import PidEpoch  # noqa: E402


PHASE_ORDER = (
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
)
DERIVED_ARTIFACTS = frozenset(
    {
        "sampling-results.json",
        "cancel-results.json",
        "prefill-latency-results.json",
        "api-contract-results.json",
        "openwebui-smoke.json",
        "soak-results.json",
        "release-matrix.json",
        "summary.md",
        "SHA256SUMS",
    }
)
MAX_PNG_BYTES = 128 << 20
SECRET_MIN_BYTES = 16
SECRET_MAX_BYTES = 4096
SECRET_READ_MAX_BYTES = SECRET_MAX_BYTES + 1
SECRET_SCAN_JSON_MAX_BYTES = 16 << 20
SECRET_SCAN_MAX_NODES = 100_000
SECRET_COPY_CHUNK_BYTES = 64 << 10


class FullCampaignError(RuntimeError):
    """A campaign failed before its evidence directory could be published."""


def fail(message: str) -> NoReturn:
    raise FullCampaignError(message)


@dataclasses.dataclass(frozen=True)
class _StableFileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _StableFileIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def inode_anchor(self) -> tuple[int, int, int, int, int]:
        return (self.device, self.inode, self.mode, self.uid, self.gid)


def _secret_file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for campaign secret snapshots")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _private_directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for private campaign directories")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _write_all(descriptor: int, raw: bytes, label: str) -> None:
    offset = 0
    try:
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                fail(f"{label} write made no progress")
            offset += written
    except FullCampaignError:
        raise
    except OSError:
        fail(f"failed to write {label}")


def _normalize_secret(raw: bytes, label: str) -> bytes:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if (
        not SECRET_MIN_BYTES <= len(raw) <= SECRET_MAX_BYTES
        or b"\r" in raw
        or b"\n" in raw
        or b"\x00" in raw
    ):
        fail(f"{label} must contain one bounded secret line")
    return raw


def _snapshot_secret_file(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    label: str,
) -> bytes:
    if (
        not isinstance(path, os.PathLike)
        or type(expected_uid) is not int
        or expected_uid < 0
        or type(expected_gid) is not int
        or expected_gid < 0
        or type(expected_mode) is not int
    ):
        fail(f"{label} snapshot binding differs")
    absolute = Path(os.path.abspath(path))
    descriptor = -1
    try:
        descriptor = os.open(absolute, _secret_file_flags())
        before = _StableFileIdentity.from_stat(os.fstat(descriptor))
        entry_before = _StableFileIdentity.from_stat(
            os.stat(absolute, follow_symlinks=False)
        )
        if (
            before != entry_before
            or not stat.S_ISREG(before.mode)
            or stat.S_IMODE(before.mode) != expected_mode
            or before.links != 1
            or before.uid != expected_uid
            or before.gid != expected_gid
            or not SECRET_MIN_BYTES <= before.size <= SECRET_READ_MAX_BYTES
        ):
            fail(f"{label} file identity, owner, mode, or size differs")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(SECRET_COPY_CHUNK_BYTES, SECRET_READ_MAX_BYTES + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > SECRET_READ_MAX_BYTES:
                fail(f"{label} file exceeds its byte bound")
            chunks.append(chunk)

        after = _StableFileIdentity.from_stat(os.fstat(descriptor))
        entry_after = _StableFileIdentity.from_stat(
            os.stat(absolute, follow_symlinks=False)
        )
        if before != after or before != entry_after or total != before.size:
            fail(f"{label} file changed while it was read")
        return _normalize_secret(b"".join(chunks), label)
    except FullCampaignError:
        raise
    except OSError:
        fail(f"failed to snapshot {label} without following links")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail(f"failed to close {label} snapshot")


def snapshot_api_secret(path: Path) -> bytes:
    """Snapshot the root-owned, execution-group-readable gateway credential."""

    return _snapshot_secret_file(
        path,
        expected_uid=0,
        expected_gid=os.getegid(),
        expected_mode=0o640,
        label="API credential",
    )


def snapshot_openwebui_token(path: Path) -> bytes:
    """Snapshot the private OpenWebUI bearer token owned by the execution user."""

    return _snapshot_secret_file(
        path,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_mode=0o600,
        label="OpenWebUI token",
    )


def _validate_private_parent(parent: Path, uid: int, gid: int, label: str) -> Path:
    absolute = Path(os.path.abspath(parent))
    try:
        if absolute.resolve(strict=True) != absolute:
            fail(f"{label} contains a symbolic link")
        identity = _StableFileIdentity.from_stat(absolute.lstat())
    except FullCampaignError:
        raise
    except OSError:
        fail(f"{label} is unavailable")
    if (
        not stat.S_ISDIR(identity.mode)
        or stat.S_ISLNK(identity.mode)
        or stat.S_IMODE(identity.mode) != 0o700
        or identity.uid != uid
        or identity.gid != gid
    ):
        fail(f"{label} owner or mode differs")
    return absolute


def _write_private_master(
    directory_fd: int,
    name: str,
    raw: bytes,
    *,
    uid: int,
    gid: int,
) -> _StableFileIdentity:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw, f"campaign secret master {name}")
        os.fsync(descriptor)
        identity = _StableFileIdentity.from_stat(os.fstat(descriptor))
        entry = _StableFileIdentity.from_stat(
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if (
            identity != entry
            or not stat.S_ISREG(identity.mode)
            or stat.S_IMODE(identity.mode) != 0o600
            or identity.links != 1
            or identity.uid != uid
            or identity.gid != gid
            or identity.size != len(raw)
        ):
            fail("campaign secret master identity differs")
        return identity
    except FullCampaignError:
        raise
    except OSError:
        fail("failed to create a private campaign secret master")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail("failed to close a private campaign secret master")


class CampaignSecretOwner:
    """Own two private master copies for the lifetime of one campaign."""

    _API_NAME = "gateway-api-key"
    _TOKEN_NAME = "openwebui-token"

    def __init__(
        self,
        directory: Path,
        directory_fd: int,
        directory_identity: _StableFileIdentity,
        parent: Path,
        parent_fd: int,
        api_identity: _StableFileIdentity,
        token_identity: _StableFileIdentity,
        *,
        uid: int,
        gid: int,
    ) -> None:
        self.directory = directory
        self.api_key_path = directory / self._API_NAME
        self.openwebui_token_path = directory / self._TOKEN_NAME
        self._directory_fd = directory_fd
        self._directory_identity = directory_identity
        self._parent = parent
        self._parent_fd = parent_fd
        self._file_identities = {
            self._API_NAME: api_identity,
            self._TOKEN_NAME: token_identity,
        }
        self.uid = uid
        self.gid = gid
        self.closed = False

    @classmethod
    def create(
        cls,
        api_secret: bytes,
        openwebui_token: bytes,
        *,
        parent: Path | None = None,
    ) -> CampaignSecretOwner:
        expected_uid = os.geteuid()
        expected_gid = os.getegid()
        if type(api_secret) is not bytes or type(openwebui_token) is not bytes:
            fail("campaign secret master values must be bytes")
        api_value = _normalize_secret(api_secret, "API credential")
        token_value = _normalize_secret(openwebui_token, "OpenWebUI token")
        root = _validate_private_parent(
            Path(f"/run/user/{expected_uid}") if parent is None else parent,
            expected_uid,
            expected_gid,
            "campaign secret parent",
        )

        parent_fd = -1
        directory_fd = -1
        directory: Path | None = None
        directory_name: str | None = None
        created_names: list[str] = []
        try:
            parent_fd = os.open(root, _private_directory_flags())
            parent_identity = _StableFileIdentity.from_stat(os.fstat(parent_fd))
            if parent_identity != _StableFileIdentity.from_stat(root.lstat()):
                fail("campaign secret parent changed while opening")
            for _attempt in range(16):
                candidate = f"ullm-sq8-campaign-{secrets.token_hex(12)}"
                try:
                    os.mkdir(candidate, 0o700, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                directory_name = candidate
                break
            if directory_name is None:
                fail("failed to allocate a private campaign secret directory")
            directory = root / directory_name
            directory_fd = os.open(
                directory_name, _private_directory_flags(), dir_fd=parent_fd
            )
            api_identity = _write_private_master(
                directory_fd,
                cls._API_NAME,
                api_value,
                uid=expected_uid,
                gid=expected_gid,
            )
            created_names.append(cls._API_NAME)
            token_identity = _write_private_master(
                directory_fd,
                cls._TOKEN_NAME,
                token_value,
                uid=expected_uid,
                gid=expected_gid,
            )
            created_names.append(cls._TOKEN_NAME)
            os.fsync(directory_fd)
            directory_identity = _StableFileIdentity.from_stat(os.fstat(directory_fd))
            entry = _StableFileIdentity.from_stat(
                os.stat(directory_name, dir_fd=parent_fd, follow_symlinks=False)
            )
            if (
                directory_identity != entry
                or not stat.S_ISDIR(directory_identity.mode)
                or stat.S_IMODE(directory_identity.mode) != 0o700
                or directory_identity.uid != expected_uid
                or directory_identity.gid != expected_gid
                or set(os.listdir(directory_fd)) != set(created_names)
            ):
                fail("campaign secret directory identity differs")
            if (
                _StableFileIdentity.from_stat(root.lstat()).inode_anchor()
                != parent_identity.inode_anchor()
            ):
                fail("campaign secret parent changed during master creation")
            os.fsync(parent_fd)
            return cls(
                directory,
                directory_fd,
                directory_identity,
                root,
                parent_fd,
                api_identity,
                token_identity,
                uid=expected_uid,
                gid=expected_gid,
            )
        except BaseException:
            if directory_fd >= 0:
                try:
                    entries = set(os.listdir(directory_fd))
                except OSError:
                    entries = set()
                for name in (cls._TOKEN_NAME, cls._API_NAME):
                    if name not in entries:
                        continue
                    try:
                        os.unlink(name, dir_fd=directory_fd)
                    except OSError:
                        pass
                try:
                    os.close(directory_fd)
                except OSError:
                    pass
            if directory_name is not None and parent_fd >= 0:
                try:
                    os.rmdir(directory_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if parent_fd >= 0:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass
            raise

    def __enter__(self) -> CampaignSecretOwner:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            error.add_note("campaign secret cleanup also failed")

    def revalidate(self) -> None:
        """Verify that both private masters and their directory are unchanged."""

        if self.closed:
            fail("campaign secret owner is already closed")
        try:
            directory = _StableFileIdentity.from_stat(os.fstat(self._directory_fd))
            entry = _StableFileIdentity.from_stat(
                os.stat(
                    self.directory.name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
            )
            if (
                directory != self._directory_identity
                or entry != self._directory_identity
                or set(os.listdir(self._directory_fd)) != set(self._file_identities)
            ):
                fail("campaign secret directory changed")
            for name, expected in self._file_identities.items():
                current = _StableFileIdentity.from_stat(
                    os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
                )
                if current != expected:
                    fail("campaign secret master changed")
        except FullCampaignError:
            raise
        except OSError:
            fail("campaign secret masters are unavailable")

    def close(self) -> None:
        if self.closed:
            return
        tampered = False
        cleanup_failed = False
        parent_entry_matches = False
        try:
            descriptor_identity = _StableFileIdentity.from_stat(
                os.fstat(self._directory_fd)
            )
            parent_entry = _StableFileIdentity.from_stat(
                os.stat(
                    self.directory.name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
            )
            parent_entry_matches = (
                parent_entry.inode_anchor() == self._directory_identity.inode_anchor()
            )
            entries = set(os.listdir(self._directory_fd))
            if (
                descriptor_identity != self._directory_identity
                or parent_entry != self._directory_identity
                or entries != set(self._file_identities)
            ):
                tampered = True
        except OSError:
            tampered = True
            entries = set()

        for name, expected in self._file_identities.items():
            try:
                current = _StableFileIdentity.from_stat(
                    os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
                )
                if current != expected:
                    tampered = True
                if (current.device, current.inode) == (
                    expected.device,
                    expected.inode,
                ):
                    os.unlink(name, dir_fd=self._directory_fd)
            except FileNotFoundError:
                tampered = True
            except OSError:
                cleanup_failed = True
        try:
            os.fsync(self._directory_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._directory_fd)
        except OSError:
            cleanup_failed = True
        finally:
            self._directory_fd = -1
        try:
            if parent_entry_matches:
                os.rmdir(self.directory.name, dir_fd=self._parent_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.fsync(self._parent_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._parent_fd)
        except OSError:
            cleanup_failed = True
        finally:
            self._parent_fd = -1
        self.closed = True
        if tampered:
            fail("campaign secret directory or masters changed before cleanup")
        if cleanup_failed:
            fail("failed to remove private campaign secret masters")


class CampaignSecretScanner:
    """Streaming raw-secret scanner that retains cross-chunk overlap."""

    def __init__(self, secrets: tuple[bytes, ...], label: str) -> None:
        self.secrets = secrets
        self.label = label
        self.tail = b""
        self.overlap = max(len(value) for value in secrets) - 1

    def feed(self, chunk: bytes) -> None:
        if type(chunk) is not bytes:
            fail(f"{self.label} secret scan chunk type differs")
        combined = self.tail + chunk
        if any(secret in combined for secret in self.secrets):
            fail(f"{self.label} contains forbidden campaign cleartext")
        self.tail = combined[-self.overlap :] if self.overlap else b""


class CampaignSecretGuard:
    """Scan API and browser credentials in raw, structured, and streamed evidence."""

    def __init__(self, values: Iterable[bytes]) -> None:
        secrets = tuple(values)
        if (
            not secrets
            or any(type(value) is not bytes or not value for value in secrets)
            or len(set(secrets)) != len(secrets)
        ):
            fail("campaign forbidden secret set differs")
        self.secrets = secrets

    def _reject_raw(self, raw: bytes, label: str) -> None:
        if any(secret in raw for secret in self.secrets):
            fail(f"{label} contains forbidden campaign cleartext")

    def _scan_json_if_complete(self, raw: bytes, label: str) -> bool:
        if not raw or len(raw) > SECRET_SCAN_JSON_MAX_BYTES:
            return False
        if raw.count(b",") + 1 > SECRET_SCAN_MAX_NODES:
            fail(f"{label} exceeds the semantic secret-scan node bound")
        try:
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError, RecursionError):
            return False
        self.reject_json_value(value, label)
        return True

    def reject(self, raw: bytes, label: str) -> None:
        if type(raw) is not bytes or type(label) is not str or not label:
            fail("campaign raw secret scan arguments differ")
        self._reject_raw(raw, label)
        self._scan_json_if_complete(raw, label)

    def reject_json_value(self, value: Any, label: str) -> None:
        pending: list[tuple[Any, int]] = [(value, 0)]
        visited = 0
        while pending:
            item, depth = pending.pop()
            visited += 1
            if depth > 128 or visited > SECRET_SCAN_MAX_NODES:
                fail(f"{label} exceeds the semantic secret-scan bound")
            if type(item) is dict:
                if visited + len(pending) + len(item) > SECRET_SCAN_MAX_NODES:
                    fail(f"{label} exceeds the semantic secret-scan node bound")
                for key, child in item.items():
                    if type(key) is str:
                        try:
                            self._reject_raw(
                                key.encode("utf-8", errors="strict"), label
                            )
                        except UnicodeError:
                            fail(f"{label} contains a non-UTF-8 object key")
                    pending.append((child, depth + 1))
            elif type(item) in {list, tuple}:
                if visited + len(pending) + len(item) > SECRET_SCAN_MAX_NODES:
                    fail(f"{label} exceeds the semantic secret-scan node bound")
                pending.extend((child, depth + 1) for child in item)
            elif type(item) is str:
                try:
                    self._reject_raw(item.encode("utf-8", errors="strict"), label)
                except UnicodeError:
                    fail(f"{label} contains a non-UTF-8 string")
            elif type(item) in {bytes, bytearray}:
                self._reject_raw(bytes(item), label)

    def scanner(self, label: str) -> CampaignSecretScanner:
        if type(label) is not str or not label:
            fail("campaign streaming secret scan label differs")
        return CampaignSecretScanner(self.secrets, label)

    def scan_file(self, path: Path, label: str) -> None:
        descriptor = -1
        document = bytearray()
        line = bytearray()
        scanner = self.scanner(label)
        absolute = Path(os.path.abspath(path))
        semantic_mode = absolute.suffix

        def finish_line() -> None:
            stripped = bytes(line).strip()
            if stripped and not self._scan_json_if_complete(stripped, label):
                fail(f"{label} contains a JSONL row that cannot be semantic-scanned")
            line.clear()

        try:
            descriptor = os.open(absolute, _secret_file_flags())
            before = _StableFileIdentity.from_stat(os.fstat(descriptor))
            if not stat.S_ISREG(before.mode) or before != _StableFileIdentity.from_stat(
                os.stat(absolute, follow_symlinks=False)
            ):
                fail(f"{label} scan file identity differs")
            while chunk := os.read(descriptor, SECRET_COPY_CHUNK_BYTES):
                scanner.feed(chunk)
                if semantic_mode == ".json":
                    if len(document) + len(chunk) > SECRET_SCAN_JSON_MAX_BYTES:
                        fail(f"{label} exceeds the semantic JSON scan bound")
                    document.extend(chunk)
                elif semantic_mode == ".jsonl":
                    parts = chunk.split(b"\n")
                    for index, part in enumerate(parts):
                        if len(line) + len(part) > SECRET_SCAN_JSON_MAX_BYTES:
                            fail(f"{label} JSONL row exceeds the semantic scan bound")
                        line.extend(part)
                        if index < len(parts) - 1:
                            finish_line()
            if semantic_mode == ".json":
                if not self._scan_json_if_complete(bytes(document).strip(), label):
                    fail(f"{label} is not completely semantic-scannable JSON")
            elif semantic_mode == ".jsonl" and line:
                finish_line()
            after = _StableFileIdentity.from_stat(os.fstat(descriptor))
            entry_after = _StableFileIdentity.from_stat(
                os.stat(absolute, follow_symlinks=False)
            )
            if before != after or before != entry_after:
                fail(f"{label} changed while it was scanned")
        except FullCampaignError:
            raise
        except OSError:
            fail(f"failed to scan {label} without following links")
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    fail(f"failed to close {label} after secret scanning")


class CampaignLockOwner:
    """Hold the host-wide full-campaign lock through one open description."""

    def __init__(
        self,
        path: Path,
        descriptor: int,
        parent_descriptor: int,
        identity: _StableFileIdentity,
    ):
        self.path = path
        self._descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self._identity = identity
        self.closed = False

    @classmethod
    def acquire(
        cls,
        path: Path,
    ) -> CampaignLockOwner:
        expected_uid = os.geteuid()
        expected_gid = os.getegid()
        absolute = Path(os.path.abspath(path))
        _validate_private_parent(
            absolute.parent, expected_uid, expected_gid, "campaign lock parent"
        )
        descriptor = -1
        parent_descriptor = -1
        locked = False
        try:
            parent_descriptor = os.open(absolute.parent, _private_directory_flags())
            if _StableFileIdentity.from_stat(
                os.fstat(parent_descriptor)
            ) != _StableFileIdentity.from_stat(absolute.parent.lstat()):
                fail("campaign lock parent changed while opening")
            descriptor = os.open(
                absolute.name,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            identity = _StableFileIdentity.from_stat(os.fstat(descriptor))
            entry = _StableFileIdentity.from_stat(
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            if (
                identity != entry
                or not stat.S_ISREG(identity.mode)
                or stat.S_IMODE(identity.mode) != 0o600
                or identity.links != 1
                or identity.uid != expected_uid
                or identity.gid != expected_gid
            ):
                fail("campaign lock file identity, owner, or mode differs")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fail("another full SQ8 campaign already holds the lock")
            locked = True
            locked_identity = _StableFileIdentity.from_stat(os.fstat(descriptor))
            locked_entry = _StableFileIdentity.from_stat(
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            if locked_identity != identity or locked_entry != identity:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                fail("campaign lock file changed during acquisition")
            return cls(absolute, descriptor, parent_descriptor, identity)
        except BaseException as error:
            cleanup_failed = False
            if locked and descriptor >= 0:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    cleanup_failed = True
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    cleanup_failed = True
            if parent_descriptor >= 0:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    cleanup_failed = True
            if cleanup_failed:
                error.add_note("campaign lock acquisition cleanup also failed")
            if isinstance(error, OSError):
                fail("failed to acquire the full SQ8 campaign lock")
            raise

    def __enter__(self) -> CampaignLockOwner:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            error.add_note("campaign lock cleanup also failed")

    def revalidate(self) -> None:
        if self.closed:
            fail("campaign lock owner is already closed")
        try:
            current = _StableFileIdentity.from_stat(os.fstat(self._descriptor))
            entry = _StableFileIdentity.from_stat(
                os.stat(
                    self.path.name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError:
            fail("campaign lock file is unavailable")
        if current != self._identity or entry != self._identity:
            fail("campaign lock file changed while held")

    def close(self) -> None:
        if self.closed:
            return
        identity_changed = False
        cleanup_failed = False
        try:
            current = _StableFileIdentity.from_stat(os.fstat(self._descriptor))
            entry = _StableFileIdentity.from_stat(
                os.stat(
                    self.path.name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            )
            if current != self._identity or entry != self._identity:
                identity_changed = True
        except OSError:
            identity_changed = True
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._descriptor)
        except OSError:
            cleanup_failed = True
        finally:
            self._descriptor = -1
        try:
            os.close(self._parent_descriptor)
        except OSError:
            cleanup_failed = True
        finally:
            self._parent_descriptor = -1
        self.closed = True
        if identity_changed:
            fail("campaign lock file changed while held")
        if cleanup_failed:
            fail("failed to release the full SQ8 campaign lock")


class SessionWriterProtocol(Protocol):
    """The subset of ``collect-sq8-openwebui-release.SessionWriter`` used here."""

    counts: collections.Counter[str]
    sequence: int
    writer: "AtomicJsonlWriterProtocol"

    def append(
        self, record_type: str, phase: str, case_id: str | None, **fields: Any
    ) -> None: ...


class AtomicJsonlWriterProtocol(Protocol):
    def write_value(self, value: dict[str, Any]) -> None: ...

    def commit(self) -> None: ...

    def abort_close(self) -> None: ...


class ClaimedGatewayEventProtocol(Protocol):
    phase: str
    case_id: str
    fields: dict[str, Any]


class CampaignJournalProtocol(Protocol):
    def start(self) -> str: ...

    def checkpoint(self, phase: str, deadline_ns: int) -> str: ...

    def arm_restart_transition(self) -> None: ...

    def claim_bundle_records(
        self, claims: Iterable[Any], deadline_ns: int
    ) -> list[ClaimedGatewayEventProtocol]: ...

    def confirm_restart_epoch(self, epoch: PidEpoch) -> None: ...

    def seal(self, expected_final_cursor: str, deadline_ns: int) -> str: ...

    def abort(self) -> None: ...


class ResourceSegmentResultProtocol(Protocol):
    segment: str
    identity: Any
    sampling_cases: tuple[dict[str, Any], ...]


class ResourceAdapterProtocol(Protocol):
    """A wrapper around the existing ``ResourceSegmentCollector`` instance."""

    def collect_normal(
        self, *, expected_identity: Any | None = None
    ) -> ResourceSegmentResultProtocol: ...

    def collect_restart(
        self, normal_identity: Any, *, expected_identity: Any | None = None
    ) -> ResourceSegmentResultProtocol: ...

    def close(self) -> None: ...


class ApiResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    journal_records: tuple[dict[str, Any], ...]
    quiet_check_records: tuple[dict[str, Any], ...]
    derived_view: dict[str, Any]
    final_journal_cursor: str


class CombinedResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


class DirectResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


class ScreenshotProtocol(Protocol):
    path: Path
    bytes: int
    sha256: str


class StopResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    screenshot_evidence: ScreenshotProtocol
    derived_view: dict[str, Any]


class FailureScreenshotProtocol(Protocol):
    source_path: Path
    bundle_path: str
    bytes: int
    sha256: str


class FailureResultProtocol(Protocol):
    browser_action_records: tuple[dict[str, Any], ...]
    fault_injection_record: dict[str, Any]
    lifecycle_claims: tuple[Any, ...]
    restart_probe_record: dict[str, Any]
    screenshot_evidence: FailureScreenshotProtocol
    derived_view: dict[str, Any]


class LatencyResultProtocol(Protocol):
    http_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[Any, ...]
    derived_view: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PreflightPhaseResult:
    environment_bytes: bytes
    model_identity_bytes: bytes
    header_fields: dict[str, Any]
    resource_header: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class FinalPhaseResult:
    lifecycle_probe_record: dict[str, Any]
    completed_utc: str
    completed_monotonic_ns: int
    final_git_commit: str
    final_git_status_raw: str


@dataclasses.dataclass(frozen=True)
class CampaignConfig:
    final_path: Path
    uid: int
    gid: int
    boot_id: str
    normal_epoch: PidEpoch
    operation_timeout_ns: int = 10_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.final_path, os.PathLike):
            fail("campaign final path type differs")
        if type(self.uid) is not int or self.uid < 0:
            fail("campaign UID binding differs")
        if type(self.gid) is not int or self.gid < 0:
            fail("campaign GID binding differs")
        if type(self.boot_id) is not str or len(self.boot_id) != 32:
            fail("campaign boot ID binding differs")
        try:
            int(self.boot_id, 16)
        except ValueError:
            fail("campaign boot ID syntax differs")
        if self.boot_id != self.boot_id.lower():
            fail("campaign boot ID must be lowercase")
        if not isinstance(self.normal_epoch, PidEpoch):
            fail("campaign normal PID epoch type differs")
        if type(self.operation_timeout_ns) is not int or self.operation_timeout_ns < 1:
            fail("campaign timeout binding differs")


@dataclasses.dataclass(frozen=True)
class CampaignEvidence:
    preflight: PreflightPhaseResult
    api_contract: ApiResultProtocol
    combined: CombinedResultProtocol
    direct_cancel: DirectResultProtocol
    stop: StopResultProtocol
    resource_normal: ResourceSegmentResultProtocol
    failure: FailureResultProtocol
    resource_restart: ResourceSegmentResultProtocol
    latency: LatencyResultProtocol
    final: FinalPhaseResult


@dataclasses.dataclass(frozen=True)
class RenderContext:
    stage_path: Path
    evidence: CampaignEvidence


class ViewsRenderer(Protocol):
    """Boundary implemented by ``sq8_full_campaign_views.py`` when wired."""

    def render(self, context: RenderContext) -> Mapping[str, bytes]: ...


class IndependentValidator(Protocol):
    """Validate the sealed prevalidation set and exclusively write validation."""

    def validate(self, stage_path: Path) -> FileEvidence: ...


class CampaignBackend(Protocol):
    """All command construction and live operations stay behind this boundary."""

    def now_ns(self) -> int: ...

    def scan_evidence(self, raw: bytes, label: str) -> None: ...

    def make_session_writer(self, path: Path) -> SessionWriterProtocol: ...

    def make_resource_writer(self, path: Path) -> AtomicJsonlWriterProtocol: ...

    def make_journal_capture(
        self, path: Path, boot_id: str, normal_epoch: PidEpoch
    ) -> CampaignJournalProtocol: ...

    def preflight(self, work_dir: Path) -> PreflightPhaseResult: ...

    def api_contract(self, work_dir: Path) -> ApiResultProtocol: ...

    def combined(self, work_dir: Path) -> CombinedResultProtocol: ...

    def direct_cancel(self, work_dir: Path) -> DirectResultProtocol: ...

    def stop(self, work_dir: Path) -> StopResultProtocol: ...

    def make_resource_adapter(
        self,
        *,
        normal_work_dir: Path,
        restart_work_dir: Path,
        stage_path: Path,
        session: SessionWriterProtocol,
        resource: AtomicJsonlWriterProtocol,
        journal: CampaignJournalProtocol,
    ) -> ResourceAdapterProtocol: ...

    def failure(self, work_dir: Path) -> FailureResultProtocol: ...

    def latency(self, work_dir: Path) -> LatencyResultProtocol: ...

    def final(self, work_dir: Path) -> FinalPhaseResult: ...

    def close(self) -> None: ...


BundleFactory = Callable[..., AtomicCampaignDirectory]


class _CampaignRuntimeGuard:
    """Close every live runtime owner before the bundle removes its stage."""

    def __init__(self, backend: CampaignBackend):
        self.backend = backend
        self.resource_adapter: ResourceAdapterProtocol | None = None
        self.journal: CampaignJournalProtocol | None = None
        self.session: SessionWriterProtocol | None = None
        self.resource: AtomicJsonlWriterProtocol | None = None
        self.backend_closed = False
        self.completed = False

    def __enter__(self) -> _CampaignRuntimeGuard:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        if self.completed:
            return
        cleanup_errors: list[BaseException] = []

        def attempt(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)

        if self.resource_adapter is not None:
            attempt(self.resource_adapter.close)
        if not self.backend_closed:
            attempt(self.backend.close)
        if self.resource is not None:
            attempt(self.resource.abort_close)
        if self.session is not None:
            attempt(self.session.writer.abort_close)
        if self.journal is not None:
            attempt(self.journal.abort)

        if not cleanup_errors:
            return
        if error is not None:
            error.add_note(
                "campaign cleanup also failed after every cleanup owner was attempted"
            )
            return
        raise FullCampaignError("campaign runtime cleanup failed") from cleanup_errors[
            0
        ]


def _deadline(backend: CampaignBackend, config: CampaignConfig) -> int:
    now = backend.now_ns()
    if type(now) is not int or now < 0:
        fail("campaign backend clock differs")
    return now + config.operation_timeout_ns


def _append_hook_record(
    session: SessionWriterProtocol,
    record: dict[str, Any],
    *,
    expected_phase: str,
    expected_type: str | None = None,
) -> None:
    if type(record) is not dict or set(record) != {
        "record_type",
        "phase",
        "case_id",
        "fields",
    }:
        fail("campaign hook record shape differs")
    record_type = record["record_type"]
    phase = record["phase"]
    case_id = record["case_id"]
    fields = record["fields"]
    if (
        type(record_type) is not str
        or (expected_type is not None and record_type != expected_type)
        or phase != expected_phase
        or type(case_id) is not str
        or not case_id
        or type(fields) is not dict
    ):
        fail("campaign hook record identity differs")
    session.append(record_type, phase, case_id, **fields)


def _append_claimed(
    session: SessionWriterProtocol,
    claimed: Iterable[ClaimedGatewayEventProtocol],
    *,
    expected_phase: str,
) -> None:
    for item in claimed:
        if (
            item.phase != expected_phase
            or type(item.case_id) is not str
            or not item.case_id
            or type(item.fields) is not dict
        ):
            fail("claimed campaign lifecycle identity differs")
        session.append("gateway_event", item.phase, item.case_id, **item.fields)


def _claim_phase(
    journal: CampaignJournalProtocol,
    session: SessionWriterProtocol,
    claims: Iterable[Any],
    *,
    expected_phase: str,
    deadline_ns: int,
) -> tuple[ClaimedGatewayEventProtocol, ...]:
    materialized = tuple(claims)
    claimed = tuple(journal.claim_bundle_records(materialized, deadline_ns))
    if len(claimed) != len(materialized):
        fail("campaign journal claim cardinality differs")
    _append_claimed(session, claimed, expected_phase=expected_phase)
    return claimed


def _checkpoint(
    journal: CampaignJournalProtocol,
    backend: CampaignBackend,
    config: CampaignConfig,
    phase: str,
) -> str:
    return journal.checkpoint(phase, _deadline(backend, config))


_IDENTITY_FIELDS = (
    "control_group",
    "gateway_pid",
    "gateway_starttime_ticks",
    "worker_pid",
    "worker_starttime_ticks",
    "n_restarts",
)


def _identity_values(identity: Any) -> tuple[Any, ...]:
    try:
        values = tuple(getattr(identity, field) for field in _IDENTITY_FIELDS)
    except AttributeError:
        fail("resource process identity shape differs")
    if (
        type(values[0]) is not str
        or not values[0]
        or any(type(value) is not int or value < 1 for value in values[1:5])
        or type(values[5]) is not int
        or values[5] < 0
    ):
        fail("resource process identity values differ")
    return values


def _restart_identity_from_probe(normal_identity: Any, fields: dict[str, Any]) -> Any:
    if set(_IDENTITY_FIELDS) - set(fields):
        fail("restart probe lacks a process identity")
    values = {field: fields[field] for field in _IDENTITY_FIELDS}
    try:
        restart = type(normal_identity)(**values)
    except (TypeError, ValueError):
        fail("restart probe cannot bind the resource identity type")
    normal_values = _identity_values(normal_identity)
    restart_values = _identity_values(restart)
    if (
        restart_values[0] != normal_values[0]
        or restart_values[1] == normal_values[1]
        or restart_values[3] == normal_values[3]
        or restart_values[5] != normal_values[5] + 1
    ):
        fail("restart probe process epoch differs")
    return restart


def _validate_resource_result(
    session: SessionWriterProtocol,
    result: ResourceSegmentResultProtocol,
    *,
    segment: str,
    prior_probe_count: int,
) -> None:
    if result.segment != segment:
        fail("resource adapter segment result differs")
    _identity_values(result.identity)
    try:
        sampling_cases = result.sampling_cases
    except AttributeError:
        fail("resource adapter result lacks sampling cases")
    expected_sampling_count = 20 if segment == "normal" else 0
    if (
        type(sampling_cases) is not tuple
        or len(sampling_cases) != expected_sampling_count
        or any(type(item) is not dict for item in sampling_cases)
    ):
        fail("resource adapter sampling result differs")
    if session.counts.get("lifecycle_probe", 0) != prior_probe_count + 1:
        fail("resource adapter did not append exactly one lifecycle probe")


def _validate_ready_probe_record(
    record: dict[str, Any],
    *,
    phase: str,
    name: str,
    expected_identity: Any | None = None,
) -> dict[str, Any]:
    if (
        type(record) is not dict
        or set(record) != {"record_type", "phase", "case_id", "fields"}
        or record["record_type"] != "lifecycle_probe"
        or record["phase"] != phase
        or record["case_id"] != name
        or type(record["fields"]) is not dict
    ):
        fail("campaign readiness probe record differs")
    fields = record["fields"]
    if set(fields) != {
        "probe",
        "observed_monotonic_ns",
        "service_active",
        "ready_http_status",
        *_IDENTITY_FIELDS,
    }:
        fail("campaign readiness probe fields differ")
    if (
        fields["probe"] != name
        or type(fields["observed_monotonic_ns"]) is not int
        or fields["observed_monotonic_ns"] < 0
        or fields["service_active"] is not True
        or fields["ready_http_status"] != 200
    ):
        fail("campaign readiness probe state differs")
    if expected_identity is not None and tuple(
        fields[field] for field in _IDENTITY_FIELDS
    ) != _identity_values(expected_identity):
        fail("campaign readiness probe process identity differs")
    return fields


def _copy_stop_screenshot(
    bundle: AtomicCampaignDirectory,
    result: StopResultProtocol,
    backend: CampaignBackend,
) -> None:
    evidence = result.screenshot_evidence
    bundle.copy_file(
        evidence.path,
        "browser/openwebui-stop-before.png",
        expected_bytes=evidence.bytes,
        expected_sha256=evidence.sha256,
        maximum_bytes=MAX_PNG_BYTES,
        scan=backend.scan_evidence,
    )


def _copy_failure_screenshot(
    bundle: AtomicCampaignDirectory,
    result: FailureResultProtocol,
    backend: CampaignBackend,
) -> None:
    evidence = result.screenshot_evidence
    if evidence.bundle_path != "browser/post-header-failure.png":
        fail("failure screenshot bundle path differs")
    bundle.copy_file(
        evidence.source_path,
        evidence.bundle_path,
        expected_bytes=evidence.bytes,
        expected_sha256=evidence.sha256,
        maximum_bytes=MAX_PNG_BYTES,
        scan=backend.scan_evidence,
    )


def _failure_phase(
    *,
    result: FailureResultProtocol,
    journal: CampaignJournalProtocol,
    session: SessionWriterProtocol,
    normal_identity: Any,
    backend: CampaignBackend,
    config: CampaignConfig,
) -> Any:
    actions = tuple(result.browser_action_records)
    claims = tuple(result.lifecycle_claims)
    if len(actions) != 9 or len(claims) != 10:
        fail("post-header failure action or lifecycle cardinality differs")
    if [getattr(item, "case_id", None) for item in claims] != (
        ["post-header-failure"] * 5 + ["post-header-recovery"] * 5
    ):
        fail("post-header failure lifecycle partition differs")

    for record in actions[:4]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    claimed = tuple(journal.claim_bundle_records(claims, _deadline(backend, config)))
    if len(claimed) != 10:
        fail("post-header failure global journal claim cardinality differs")
    _append_claimed(session, claimed[:4], expected_phase="post_header_failure")
    _append_hook_record(
        session,
        result.fault_injection_record,
        expected_phase="post_header_failure",
        expected_type="fault_injection",
    )
    _append_claimed(session, claimed[4:5], expected_phase="post_header_failure")
    _append_hook_record(
        session,
        actions[4],
        expected_phase="post_header_failure",
        expected_type="browser_action",
    )
    probe = result.restart_probe_record
    probe_fields = _validate_ready_probe_record(
        probe,
        phase="post_header_failure",
        name="post-header-restart-ready",
    )
    restart_identity = _restart_identity_from_probe(normal_identity, probe_fields)
    restart_epoch = PidEpoch(
        restart_identity.gateway_pid,
        restart_identity.worker_pid,
    )
    journal.confirm_restart_epoch(restart_epoch)
    _append_hook_record(
        session,
        probe,
        expected_phase="post_header_failure",
        expected_type="lifecycle_probe",
    )
    for record in actions[5:7]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    _append_claimed(session, claimed[5:], expected_phase="post_header_failure")
    for record in actions[7:]:
        _append_hook_record(
            session,
            record,
            expected_phase="post_header_failure",
            expected_type="browser_action",
        )
    return restart_identity


def _render_artifacts(
    bundle: AtomicCampaignDirectory,
    renderer: ViewsRenderer,
    context: RenderContext,
    backend: CampaignBackend,
) -> None:
    rendered = renderer.render(context)
    if type(rendered) is not dict or set(rendered) != set(DERIVED_ARTIFACTS):
        fail("full campaign rendered artifact set differs")
    for relative in sorted(rendered, key=lambda item: item.encode("utf-8")):
        raw = rendered[relative]
        if type(raw) is not bytes or not raw:
            fail("full campaign rendered artifact bytes differ")
        bundle.write_bytes(relative, raw, scan=backend.scan_evidence)


def _seal_private_runtime_artifact(path: Path, *, uid: int, gid: int) -> None:
    """Tighten files produced by the existing collectors to the bundle's 0600 mode."""

    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or before.st_size < 1
        ):
            fail("campaign runtime artifact identity differs")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        entry = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (entry.st_dev, entry.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_nlink != 1
            or after.st_uid != uid
            or after.st_gid != gid
        ):
            fail("campaign runtime artifact changed while sealing")
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except FullCampaignError:
        raise
    except OSError:
        fail("failed to seal a campaign runtime artifact")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                fail("failed to close a campaign runtime artifact")


def run_full_campaign(
    config: CampaignConfig,
    backend: CampaignBackend,
    renderer: ViewsRenderer,
    validator: IndependentValidator,
    *,
    bundle_factory: BundleFactory = AtomicCampaignDirectory,
) -> Path:
    """Run one serial campaign and publish only independently validated evidence."""

    cleanup = _CampaignRuntimeGuard(backend)
    try:
        with (
            bundle_factory(
                config.final_path,
                uid=config.uid,
                gid=config.gid,
            ) as bundle,
            cleanup,
        ):
            session = backend.make_session_writer(
                bundle.artifact_path("raw-session-results.jsonl")
            )
            cleanup.session = session
            resource = backend.make_resource_writer(
                bundle.artifact_path("soak-resources.raw.jsonl")
            )
            cleanup.resource = resource
            journal = backend.make_journal_capture(
                bundle.artifact_path("service-journal.raw.jsonl"),
                config.boot_id,
                config.normal_epoch,
            )
            cleanup.journal = journal
            journal.start()

            preflight = backend.preflight(bundle.component_directory("preflight"))
            if not isinstance(preflight, PreflightPhaseResult):
                fail("preflight result type differs")
            bundle.write_bytes(
                "environment.json",
                preflight.environment_bytes,
                scan=backend.scan_evidence,
            )
            bundle.write_bytes(
                "model-identity.json",
                preflight.model_identity_bytes,
                scan=backend.scan_evidence,
            )
            session.append("header", "preflight", None, **preflight.header_fields)
            resource.write_value(preflight.resource_header)
            _checkpoint(journal, backend, config, "preflight")

            api = backend.api_contract(bundle.component_directory("api-contract"))
            for record in api.http_records:
                _append_hook_record(session, record, expected_phase="api_contract")
            for record in api.journal_records:
                _append_hook_record(
                    session,
                    record,
                    expected_phase="api_contract",
                    expected_type="api_journal_observation",
                )
            for record in api.quiet_check_records:
                _append_hook_record(
                    session,
                    record,
                    expected_phase="api_contract",
                    expected_type="lifecycle_quiet_check",
                )
            api_cursor = _checkpoint(journal, backend, config, "api_contract")
            if api.final_journal_cursor != api_cursor:
                fail("API contract journal boundary differs from the global campaign")

            combined = backend.combined(bundle.component_directory("combined"))
            for record in combined.browser_action_records:
                _append_hook_record(session, record, expected_phase="openwebui")
            _claim_phase(
                journal,
                session,
                combined.lifecycle_claims,
                expected_phase="openwebui",
                deadline_ns=_deadline(backend, config),
            )
            _checkpoint(journal, backend, config, "openwebui")

            direct = backend.direct_cancel(bundle.component_directory("direct-cancel"))
            for record in direct.http_records:
                _append_hook_record(session, record, expected_phase="cancellation")
            _claim_phase(
                journal,
                session,
                direct.lifecycle_claims,
                expected_phase="cancellation",
                deadline_ns=_deadline(backend, config),
            )
            stop = backend.stop(bundle.component_directory("stop"))
            for record in stop.browser_action_records:
                _append_hook_record(session, record, expected_phase="cancellation")
            _claim_phase(
                journal,
                session,
                stop.lifecycle_claims,
                expected_phase="cancellation",
                deadline_ns=_deadline(backend, config),
            )
            _copy_stop_screenshot(bundle, stop, backend)
            _checkpoint(journal, backend, config, "cancellation")

            normal_work = bundle.component_directory("resource-normal")
            failure_work = bundle.component_directory("failure")
            restart_work = bundle.component_directory("resource-restart")
            resource_adapter = backend.make_resource_adapter(
                normal_work_dir=normal_work,
                restart_work_dir=restart_work,
                stage_path=bundle.stage_path,
                session=session,
                resource=resource,
                journal=journal,
            )
            cleanup.resource_adapter = resource_adapter
            probe_count = session.counts.get("lifecycle_probe", 0)
            normal_resource = resource_adapter.collect_normal()
            _validate_resource_result(
                session,
                normal_resource,
                segment="normal",
                prior_probe_count=probe_count,
            )
            normal_values = _identity_values(normal_resource.identity)
            if normal_values[1] != config.normal_epoch.gateway_pid or normal_values[
                3
            ] != (config.normal_epoch.worker_pid):
                fail("normal resource identity differs from the journal epoch")
            _checkpoint(journal, backend, config, "resource_normal")
            journal.arm_restart_transition()

            failure = backend.failure(failure_work)
            _copy_failure_screenshot(bundle, failure, backend)
            restart_identity = _failure_phase(
                result=failure,
                journal=journal,
                session=session,
                normal_identity=normal_resource.identity,
                backend=backend,
                config=config,
            )
            _checkpoint(journal, backend, config, "post_header_failure")

            probe_count = session.counts.get("lifecycle_probe", 0)
            restart_resource = resource_adapter.collect_restart(
                normal_resource.identity,
                expected_identity=restart_identity,
            )
            _validate_resource_result(
                session,
                restart_resource,
                segment="restart",
                prior_probe_count=probe_count,
            )
            if _identity_values(restart_resource.identity) != _identity_values(
                restart_identity
            ):
                fail("restart resource identity differs from the confirmed epoch")
            resource_adapter.close()
            cleanup.resource_adapter = None
            _checkpoint(journal, backend, config, "resource_restart")

            latency = backend.latency(bundle.component_directory("latency"))
            for record in latency.http_records:
                _append_hook_record(session, record, expected_phase="latency")
            _claim_phase(
                journal,
                session,
                latency.lifecycle_claims,
                expected_phase="latency",
                deadline_ns=_deadline(backend, config),
            )
            _checkpoint(journal, backend, config, "latency")

            final = backend.final(bundle.component_directory("final"))
            backend.close()
            cleanup.backend_closed = True
            final_probe_fields = _validate_ready_probe_record(
                final.lifecycle_probe_record,
                phase="final",
                name="final-service-ready",
                expected_identity=restart_identity,
            )
            if (
                type(final.completed_utc) is not str
                or not final.completed_utc
                or type(final.completed_monotonic_ns) is not int
                or final.completed_monotonic_ns
                < final_probe_fields["observed_monotonic_ns"]
                or type(final.final_git_commit) is not str
                or len(final.final_git_commit) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in final.final_git_commit
                )
            ):
                fail("final campaign metadata differs")
            _append_hook_record(
                session,
                final.lifecycle_probe_record,
                expected_phase="final",
                expected_type="lifecycle_probe",
            )
            if session.counts.get("lifecycle_probe", 0) != 4:
                fail("full campaign must contain exactly four lifecycle probes")
            final_cursor = _checkpoint(journal, backend, config, "final")
            counts = dict(session.counts)
            counts["run_end"] = counts.get("run_end", 0) + 1
            status_raw = final.final_git_status_raw
            if type(status_raw) is not str:
                fail("final Git status type differs")
            session.append(
                "run_end",
                "final",
                None,
                completed_utc=final.completed_utc,
                completed_monotonic_ns=final.completed_monotonic_ns,
                final_git_commit=final.final_git_commit,
                final_git_status_raw=status_raw,
                final_git_status_sha256=hashlib.sha256(
                    status_raw.encode("utf-8", errors="strict")
                ).hexdigest(),
                record_counts=counts,
                final_journal_cursor=final_cursor,
            )
            resource.commit()
            session.writer.commit()
            journal.seal(final_cursor, _deadline(backend, config))
            for relative in (
                "raw-session-results.jsonl",
                "soak-resources.raw.jsonl",
                "service-journal.raw.jsonl",
                "amd-smi-metric-normal-before.json",
                "amd-smi-metric-normal-after.json",
                "amd-smi-metric-restart-before.json",
                "amd-smi-metric-restart-after.json",
            ):
                _seal_private_runtime_artifact(
                    bundle.artifact_path(relative), uid=config.uid, gid=config.gid
                )

            evidence = CampaignEvidence(
                preflight,
                api,
                combined,
                direct,
                stop,
                normal_resource,
                failure,
                restart_resource,
                latency,
                final,
            )
            _render_artifacts(
                bundle,
                renderer,
                RenderContext(bundle.stage_path, evidence),
                backend,
            )
            if set(PREVALIDATION_ROOT_FILES) - {
                path.name for path in bundle.stage_path.iterdir() if path.is_file()
            }:
                fail("full campaign prevalidation artifacts are incomplete")
            bundle.clear_component_work()
            bundle.validate_before_independent_validator()
            validation_evidence = validator.validate(bundle.stage_path)
            published = bundle.publish(validation_evidence)
            if not isinstance(published, os.PathLike):
                fail("campaign publication path type differs")
            cleanup.completed = True
            return Path(published)
    except BaseException:
        raise


@dataclasses.dataclass(frozen=True, slots=True)
class ProductionPreparationRequest:
    expected_commit: str
    expected_worker_binary_sha256: str
    run_id: str
    final_path: Path
    api_key_file: Path
    openwebui_token_file: Path


class ProductionPreparationRuntime(Protocol):
    """Dependency-injected boundary for the read-only production preparation."""

    def validate_request(self, request: ProductionPreparationRequest) -> None: ...

    def acquire_lock(self) -> Any: ...

    def capture_git_anchor(self, expected_commit: str) -> Any: ...

    def create_head_tools(self, anchor: Any) -> Any: ...

    def validate_promotion(self, anchor: Any, tools: Any) -> dict[str, Any]: ...

    def snapshot_api_key(self, path: Path) -> bytes: ...

    def snapshot_token(self, path: Path) -> bytes: ...

    def create_secret_owner(
        self, api_key: bytes, token: bytes
    ) -> CampaignSecretOwner: ...

    def build_identity(
        self,
        request: ProductionPreparationRequest,
        anchor: Any,
        promotion: dict[str, Any],
        guard: CampaignSecretGuard,
    ) -> Any: ...

    def discover_container(self) -> Any: ...

    def run_operational(self, identity: Any, container: Any) -> Any: ...

    def build_resource(
        self,
        request: ProductionPreparationRequest,
        identity: Any,
        guard: CampaignSecretGuard,
    ) -> Any: ...

    def build_config(
        self, request: ProductionPreparationRequest, identity: Any, resource: Any
    ) -> CampaignConfig: ...

    def revalidate_prepared(self, prepared: "PreparedProductionCampaign") -> None: ...


def _validate_final_destination(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        fail("campaign final destination must be an absolute Path")
    if Path(os.path.abspath(path)) != path or path.name in {"", ".", ".."}:
        fail("campaign final destination must be lexically canonical")
    try:
        parent = path.parent.lstat()
    except OSError:
        fail("campaign final destination parent is unavailable")
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        fail("campaign final destination parent is not a directory")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        fail("campaign final destination cannot be inspected")
    fail("campaign final destination already exists")


@dataclasses.dataclass(slots=True)
class PreparedProductionCampaign:
    """Own every preflight pin until publication or explicit close."""

    request: ProductionPreparationRequest
    runtime: ProductionPreparationRuntime
    lock_owner: Any
    git_anchor: Any
    tool_owner: Any
    secret_owner: CampaignSecretOwner
    secret_guard: CampaignSecretGuard
    promotion: dict[str, Any]
    identity: Any
    operational: Any
    resource: Any
    config: CampaignConfig
    closed: bool = False

    @classmethod
    def create(
        cls,
        request: ProductionPreparationRequest,
        runtime: ProductionPreparationRuntime,
    ) -> "PreparedProductionCampaign":
        runtime.validate_request(request)
        lock_owner: Any | None = None
        tool_owner: Any | None = None
        secret_owner: CampaignSecretOwner | None = None
        try:
            lock_owner = runtime.acquire_lock()
            anchor = runtime.capture_git_anchor(request.expected_commit)
            tool_owner = runtime.create_head_tools(anchor)
            promotion = runtime.validate_promotion(anchor, tool_owner)
            api_key = runtime.snapshot_api_key(request.api_key_file)
            token = runtime.snapshot_token(request.openwebui_token_file)
            try:
                secret_owner = runtime.create_secret_owner(api_key, token)
                guard = CampaignSecretGuard((api_key, token))
            finally:
                api_key = b""
                token = b""
            identity = runtime.build_identity(request, anchor, promotion, guard)
            container = runtime.discover_container()
            operational = runtime.run_operational(identity, container)
            resource = runtime.build_resource(request, identity, guard)
            config = runtime.build_config(request, identity, resource)
            return cls(
                request,
                runtime,
                lock_owner,
                anchor,
                tool_owner,
                secret_owner,
                guard,
                promotion,
                identity,
                operational,
                resource,
                config,
            )
        except BaseException as error:
            for owner, note in (
                (secret_owner, "campaign secret cleanup also failed"),
                (tool_owner, "HEAD tool cleanup also failed"),
                (lock_owner, "campaign lock cleanup also failed"),
            ):
                if owner is None:
                    continue
                try:
                    owner.close()
                except BaseException:
                    error.add_note(note)
            raise

    def revalidate(self) -> None:
        if self.closed:
            fail("prepared production campaign is already closed")
        self.lock_owner.revalidate()
        self.secret_owner.revalidate()
        self.tool_owner.revalidate()
        self.git_anchor.revalidate()
        _validate_final_destination(self.request.final_path)
        self.runtime.revalidate_prepared(self)

    def report(self) -> dict[str, Any]:
        report = {
            "schema_version": "ullm.sq8.production_preflight.v1",
            "status": "ready",
            "run_id": self.request.run_id,
            "git_commit": self.request.expected_commit,
            "worker_binary_sha256": self.request.expected_worker_binary_sha256,
            "service_epoch": {
                "gateway_pid": self.config.normal_epoch.gateway_pid,
                "worker_pid": self.config.normal_epoch.worker_pid,
            },
        }
        encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        self.secret_guard.reject(encoded, "production preflight report")
        if len(encoded) > 4096:
            fail("production preflight report exceeds its byte bound")
        return report

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        for owner in (self.secret_owner, self.tool_owner, self.lock_owner):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
        self.closed = True
        if errors:
            primary = errors[0]
            for _error in errors[1:]:
                primary.add_note("a later prepared-owner cleanup also failed")
            raise primary

    def __enter__(self) -> "PreparedProductionCampaign":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            error.add_note("prepared production campaign cleanup also failed")


def _load_hyphenated_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail("production preflight dependency cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SystemProductionPreparationRuntime:
    """Fixed-path implementation composed exclusively from existing read-only APIs."""

    def __init__(self) -> None:
        import sq8_full_campaign_identity as identity
        import sq8_full_campaign_operational as operational
        import sq8_full_campaign_prepare as prepare
        import sq8_full_campaign_production as production
        import sq8_full_campaign_resource as resource

        self.identity_module = identity
        self.operational_module = operational
        self.prepare_module = prepare
        self.production_module = production
        self.resource_module = resource
        self.settings = production.production_preflight_settings()
        self.collector = _load_hyphenated_module(
            "_sq8_full_campaign_preflight_collector",
            TOOLS_DIR / "collect-sq8-openwebui-release.py",
        )
        self.validator = _load_hyphenated_module(
            "_sq8_full_campaign_preflight_validator",
            TOOLS_DIR / "validate-sq8-openwebui-release.py",
        )

    def validate_request(self, request: ProductionPreparationRequest) -> None:
        if not isinstance(request, ProductionPreparationRequest):
            fail("production preparation request type differs")
        if (
            len(request.expected_commit) != 40
            or any(value not in "0123456789abcdef" for value in request.expected_commit)
            or len(request.expected_worker_binary_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in request.expected_worker_binary_sha256
            )
            or not request.run_id
            or len(request.run_id) > 128
        ):
            fail("production preparation anchor or run ID differs")
        _validate_final_destination(request.final_path)

    def acquire_lock(self) -> CampaignLockOwner:
        return CampaignLockOwner.acquire(
            self.production_module.canonical_campaign_lock_path()
        )

    def capture_git_anchor(self, expected_commit: str) -> Any:
        return self.production_module.GitAnchor.capture(
            self.settings, expected_commit=expected_commit
        )

    def create_head_tools(self, anchor: Any) -> Any:
        return self.production_module.HeadPromotionToolSnapshotOwner.create(
            self.settings, anchor
        )

    def validate_promotion(self, anchor: Any, tools: Any) -> dict[str, Any]:
        return self.production_module.run_pinned_full_promotion_validation(
            self.settings, anchor, tools
        )

    def snapshot_api_key(self, path: Path) -> bytes:
        return snapshot_api_secret(path)

    def snapshot_token(self, path: Path) -> bytes:
        return snapshot_openwebui_token(path)

    def create_secret_owner(self, api_key: bytes, token: bytes) -> CampaignSecretOwner:
        return CampaignSecretOwner.create(
            api_key, token, parent=self.settings.private_runtime_parent
        )

    def build_identity(
        self,
        request: ProductionPreparationRequest,
        anchor: Any,
        promotion: dict[str, Any],
        guard: CampaignSecretGuard,
    ) -> Any:
        return self.prepare_module.build_production_identity_preflight(
            anchor,
            promotion,
            expected_worker_binary_sha256=request.expected_worker_binary_sha256,
            captured_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            forbidden_values=guard.secrets,
            identity_probe=self.identity_module.SystemIdentityProbe(),
            independent_validator=self.validator,
        )

    def discover_container(self) -> Any:
        commands = self.operational_module.BoundedReadOnlyCommandReader(
            self.operational_module.production_container_discovery_commands()
        )
        return self.operational_module.discover_production_openwebui_container(commands)

    def run_operational(self, identity: Any, container: Any) -> Any:
        expectation = self.prepare_module.build_operational_expectation(
            identity.live_identity, container_id=container.container_id
        )
        commands = self.operational_module.BoundedReadOnlyCommandReader(
            self.operational_module.production_read_only_commands(expectation)
        )
        dependencies = self.operational_module.OperationalDependencies(
            commands=commands,
            http=self.operational_module.BoundedHttpReader(
                frozenset({expectation.openwebui_health_url})
            ),
            gateway_http=self.operational_module.ProductionGatewayNamespaceReader(),
            observer_paths=self.operational_module.OsObserverPathReader(),
            gpu=self.operational_module.load_worker_acceptance_gpu_reader(commands),
            monotonic_ns=time.monotonic_ns,
        )
        return self.operational_module.run_operational_preflight(
            expectation, dependencies
        )

    def build_resource(
        self,
        request: ProductionPreparationRequest,
        identity: Any,
        guard: CampaignSecretGuard,
    ) -> Any:
        started_utc = identity.identity_artifacts.environment["captured_utc"]
        return self.resource_module.build_resource_contract(
            identity.identity_artifacts,
            identity.independent_identity,
            self.validator,
            run_id=request.run_id,
            started_utc=started_utc,
            negative_case_type=self.collector.NegativeCase,
            resource_config_type=self.collector.ResourceSegmentConfig,
            forbidden_values=guard.secrets,
        )

    def build_config(
        self, request: ProductionPreparationRequest, identity: Any, resource: Any
    ) -> CampaignConfig:
        del resource
        boot_id = identity.identity_artifacts.environment["host"]["boot_id"]
        return CampaignConfig(
            request.final_path,
            os.geteuid(),
            os.getegid(),
            boot_id,
            identity.service_epoch,
        )

    def revalidate_prepared(self, prepared: PreparedProductionCampaign) -> None:
        # Promotion payload hashing is deliberately cached; all mutable live pins are
        # checked again through identity, operational, and resource composition.
        captured_utc = prepared.identity.identity_artifacts.environment["captured_utc"]
        rebuilt_identity = self.prepare_module.build_production_identity_preflight(
            prepared.git_anchor,
            prepared.promotion,
            expected_worker_binary_sha256=(
                prepared.request.expected_worker_binary_sha256
            ),
            captured_utc=captured_utc,
            forbidden_values=prepared.secret_guard.secrets,
            identity_probe=self.identity_module.SystemIdentityProbe(),
            independent_validator=self.validator,
        )
        if rebuilt_identity != prepared.identity:
            fail("production identity preflight changed")
        rebuilt_container = self.discover_container()
        rebuilt_operational = self.run_operational(rebuilt_identity, rebuilt_container)
        cached_operational = prepared.operational
        if (
            rebuilt_operational.systemd.stable_identity()
            != cached_operational.systemd.stable_identity()
            or rebuilt_operational.container != cached_operational.container
            or rebuilt_operational.gpu.stable_identity()
            != cached_operational.gpu.stable_identity()
            or rebuilt_operational.observer_parent != cached_operational.observer_parent
            or rebuilt_operational.gateway_ready != cached_operational.gateway_ready
            or rebuilt_operational.openwebui_health
            != cached_operational.openwebui_health
        ):
            fail("production operational preflight changed")
        rebuilt_resource = self.build_resource(
            prepared.request, rebuilt_identity, prepared.secret_guard
        )
        if rebuilt_resource != prepared.resource:
            fail("production resource contract changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final-path", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--openwebui-token-file", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: ProductionPreparationRuntime | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.execute:
        print(
            "full campaign production backend is not wired; refusing to execute",
            file=sys.stderr,
        )
        return 2
    request = ProductionPreparationRequest(
        arguments.expected_commit,
        arguments.expected_worker_binary_sha256,
        arguments.run_id,
        arguments.final_path,
        arguments.api_key_file,
        arguments.openwebui_token_file,
    )
    try:
        selected = SystemProductionPreparationRuntime() if runtime is None else runtime
        with PreparedProductionCampaign.create(request, selected) as prepared:
            prepared.revalidate()
            report = prepared.report()
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except KeyboardInterrupt:
        print("production preflight interrupted", file=sys.stderr)
        return 130
    except BaseException:
        print("production preflight failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignLockOwner",
    "CampaignBackend",
    "CampaignConfig",
    "CampaignEvidence",
    "CampaignSecretGuard",
    "CampaignSecretOwner",
    "CampaignSecretScanner",
    "PreparedProductionCampaign",
    "ProductionPreparationRequest",
    "ProductionPreparationRuntime",
    "SystemProductionPreparationRuntime",
    "FinalPhaseResult",
    "FileEvidence",
    "FullCampaignError",
    "IndependentValidator",
    "PHASE_ORDER",
    "PreflightPhaseResult",
    "RenderContext",
    "ResourceAdapterProtocol",
    "ViewsRenderer",
    "run_full_campaign",
    "snapshot_api_secret",
    "snapshot_openwebui_token",
]
