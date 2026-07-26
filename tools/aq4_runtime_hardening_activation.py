#!/usr/bin/env python3
"""Locked AQ4_0-to-AQ4_0 runtime-hardening activation control path.

This module deliberately does not import or call the SQ8 final-activation
route.  It owns a narrower transaction whose only supported transition is an
AQ4_0 manifest to another AQ4_0 manifest with the exact same worker hash.
All production-facing records are canonical JSON, root-owned immutable files
published with ``renameat2(RENAME_NOREPLACE)``.
"""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


PLAN_SCHEMA = "ullm.aq4_runtime_hardening_activation_plan.v2"
OPERATIONS_SCHEMA = "ullm.aq4_runtime_hardening_activation_operations.v2"
INTENT_SCHEMA = "ullm.aq4_runtime_hardening_activation_intent.v1"
OUTCOME_SCHEMA = "ullm.aq4_runtime_hardening_activation_outcome.v1"
RECOVERY_SCHEMA = "ullm.aq4_runtime_hardening_activation_recovery.v1"
ROLLBACK_SCHEMA = "ullm.aq4_runtime_hardening_rollback_outcome.v1"
ATTEMPT_SCHEMA = "ullm.aq4_runtime_hardening_recovery_attempt.v1"
LIVE_PROOF_SCHEMA = "ullm.aq4_runtime_hardening_live_proof.v3"
LIVE_PROOF_AUDIT_SCHEMA = "ullm.aq4_runtime_hardening_live_proof_audit.v1"
ISOLATED_PREFLIGHT_SCHEMA = "ullm.aq4_runtime_hardening_isolated_preflight.v1"
PREFLIGHT_SCHEMA = "ullm.aq4_runtime_hardening_activation_preflight.v2"
PREPLAN_PREFLIGHT_SCHEMA = "ullm.aq4_runtime_hardening_preplan_preflight.v1"

SERVED_MODEL_SCHEMA = "ullm.served_model.v2"
AQ4_FORMAT_ID = "AQ4_0"
AQ4_MODEL_ID = "ullm-qwen3.5-9b-aq4"
WORKER_PROTOCOL = "ullm.worker.v2"
EXPECTED_AQ4_WORKER_SHA256 = (
    "1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b"
)
DEFAULT_LOCK_PATH = Path("/etc/ullm/served-models/.active.json.activation.lock")
DEFAULT_SERVICE_UNIT = "ullm-openai.service"

ACTIVATION_CONFIRMATION = "ACTIVATE AQ4_RUNTIME_HARDENING"
ROLLBACK_CONFIRMATION = "ROLLBACK AQ4_RUNTIME_HARDENING"
RECOVERY_CONFIRMATION = "RECOVER AQ4_RUNTIME_HARDENING"

RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
LIVE_ENDPOINTS = (
    "gateway_health",
    "gateway_ready",
    "gateway_models",
    "openwebui_health",
    "openwebui_models",
)
STAGES = (
    "candidate_reconcile",
    "candidate_live_proof",
    "rollback_reconcile",
    "rollback_live_proof",
)
ISOLATED_PREFLIGHT_STAGE = "candidate_isolated_preflight"
OPERATION_STAGES = (*STAGES, ISOLATED_PREFLIGHT_STAGE)
CONTROL_TOOL_RELATIVE_PATHS = (
    Path("tools/aq4_runtime_hardening_activation.py"),
    Path("tools/aq4_runtime_hardening_operation.py"),
    Path("tools/prepare-aq4-runtime-hardening-activation.py"),
    Path("tools/run-aq4-runtime-hardening-activation.py"),
    Path("tools/rollback-aq4-runtime-hardening-activation.py"),
)


class ActivationError(RuntimeError):
    """The control route rejected an unsafe or incomplete transition."""


class ImmutablePublicationCommittedError(ActivationError):
    """A no-replace rename happened, but a later durability check failed."""


class AtomicExchangeCommittedError(ActivationError):
    """An exchange happened, but its post-exchange verification failed."""


class LiveProofFailure(ActivationError):
    """A live operation failed with a sanitized immutable-audit payload."""

    def __init__(self, stage: str, document: dict[str, Any]) -> None:
        super().__init__(f"AQ4 live proof {stage} failed")
        self.stage = stage
        self.document = document


@dataclasses.dataclass(frozen=True)
class FileIdentity:
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
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
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


@dataclasses.dataclass(frozen=True)
class Snapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: FileIdentity


@dataclasses.dataclass(frozen=True)
class PlanRecord:
    snapshot: Snapshot
    document: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    path: Path
    sha256: str
    status: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
FaultHook = Callable[[str], None]
Clock = Callable[[], datetime]


def fail(message: str) -> NoReturn:
    raise ActivationError(message)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        fail("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ActivationError("record cannot be encoded as canonical JSON") from error


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON contains duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("JSON contains a non-finite number")


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ActivationError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        fail(f"{label} root is not an object")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        fail(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        fail(f"{label} is not a Git commit")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        fail(f"{label} is invalid")
    return value


def _absolute(value: Path | str, label: str, *, exists: bool | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} is not absolute")
    if any(part in {"", ".", ".."} for part in path.parts):
        fail(f"{label} is unsafe")
    if exists is True and not path.exists():
        fail(f"{label} is unavailable")
    if exists is False and (path.exists() or path.is_symlink()):
        fail(f"{label} already exists")
    return path


def _same_identity(left: FileIdentity, right: FileIdentity) -> bool:
    return left == right


def _read_all(descriptor: int, maximum: int, *, allow_empty: bool = False) -> bytes:
    raw = bytearray()
    while len(raw) <= maximum:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
    if (not raw and not allow_empty) or len(raw) > maximum:
        fail("file exceeds its byte bound")
    return bytes(raw)


def _snapshot(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_CAPTURE_BYTES,
    required_uid: int,
    immutable: bool,
    executable: bool = False,
    allow_empty: bool = False,
) -> Snapshot:
    """Open once with O_NOFOLLOW and reject namespace or inode races."""

    path = _absolute(path, label, exists=True)
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode):
            fail(f"{label} is not a regular file")
        if before.uid != required_uid or before.links != 1:
            fail(f"{label} ownership or link count differs")
        mode = stat.S_IMODE(before.mode)
        if immutable:
            if mode != 0o444:
                fail(f"{label} is not mode 0444")
        elif mode & 0o022:
            fail(f"{label} is writable by group or other")
        if executable and mode != 0o555:
            fail(f"{label} is not mode 0555")
        raw = _read_all(descriptor, maximum, allow_empty=allow_empty)
        after = FileIdentity.from_stat(os.fstat(descriptor))
        named = FileIdentity.from_stat(path.lstat())
        if not _same_identity(before, after) or not _same_identity(after, named):
            fail(f"{label} changed while being read")
        return Snapshot(path, raw, _sha256(raw), after)
    except OSError as error:
        raise ActivationError(f"failed to read {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _seal(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "path": os.fspath(snapshot.path),
        "sha256": snapshot.sha256,
        "bytes": snapshot.identity.size,
        "mode": stat.S_IMODE(snapshot.identity.mode),
        "uid": snapshot.identity.uid,
        "gid": snapshot.identity.gid,
        "nlink": snapshot.identity.links,
        "device": snapshot.identity.device,
        "inode": snapshot.identity.inode,
        "mtime_ns": snapshot.identity.mtime_ns,
        "ctime_ns": snapshot.identity.ctime_ns,
    }


def _verify_seal(
    seal: dict[str, Any],
    label: str,
    *,
    required_uid: int,
    immutable: bool,
    executable: bool = False,
) -> Snapshot:
    fields = {
        "path",
        "sha256",
        "bytes",
        "mode",
        "uid",
        "gid",
        "nlink",
        "device",
        "inode",
        "mtime_ns",
        "ctime_ns",
    }
    _exact(seal, fields, label)
    snapshot = _snapshot(
        _absolute(seal["path"], f"{label}.path", exists=True),
        label,
        required_uid=required_uid,
        immutable=immutable,
        executable=executable,
    )
    if _seal(snapshot) != seal:
        fail(f"{label} seal drifted")
    return snapshot


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _walk_directory(path: Path) -> int:
    path = _absolute(path, "directory", exists=True)
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, _directory_flags())
        for component in path.parts[1:]:
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _open_parent(path: Path, label: str, *, required_uid: int) -> int:
    path = _absolute(path, label)
    descriptor = -1
    verification = -1
    try:
        descriptor = _walk_directory(path.parent)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            fail(f"{label} parent is unsafe")
        verification = _walk_directory(path.parent)
        if FileIdentity.from_stat(metadata) != FileIdentity.from_stat(
            os.fstat(verification)
        ):
            fail(f"{label} parent changed while being opened")
        return descriptor
    except ActivationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ActivationError(f"{label} parent is unavailable or symlinked") from error
    finally:
        if verification >= 0:
            os.close(verification)


def _renameat2(parent_fd: int, source: str, destination: str, flags: int) -> None:
    if not source or not destination or "/" in source or "/" in destination:
        fail("renameat2 names are invalid")
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, "renameat2", None)
    if operation is None:
        fail("renameat2 is unavailable")
    operation.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = operation(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    try:
        _renameat2(parent_fd, source, destination, RENAME_NOREPLACE)
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            fail("immutable output already exists")
        raise ActivationError("immutable output no-replace rename failed") from error


def _entry_raw(parent_fd: int, name: str, label: str, *, maximum: int = MAX_CAPTURE_BYTES) -> bytes:
    if not name or "/" in name:
        fail(f"{label} name is invalid")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        before = FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before.links != 1:
            fail(f"{label} is not a single-link regular file")
        raw = _read_all(descriptor, maximum)
        after = FileIdentity.from_stat(os.fstat(descriptor))
        named = FileIdentity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if before != after or after != named:
            fail(f"{label} changed while being read")
        return raw
    except OSError as error:
        raise ActivationError(f"{label} is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _committed_immutable(
    path: Path,
    raw: bytes,
    *,
    required_uid: int,
) -> Snapshot | None:
    try:
        snapshot = _snapshot(
            path,
            "committed immutable output",
            maximum=MAX_DOCUMENT_BYTES,
            required_uid=required_uid,
            immutable=True,
        )
        if snapshot.raw != raw:
            return None
        parent_fd = _open_parent(path, "committed immutable output", required_uid=required_uid)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        repeated = _snapshot(
            path,
            "committed immutable output",
            maximum=MAX_DOCUMENT_BYTES,
            required_uid=required_uid,
            immutable=True,
        )
        if repeated.raw != raw or repeated.identity != snapshot.identity:
            return None
        return repeated
    except (ActivationError, OSError):
        return None


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            fail("immutable output write made no progress")
        view = view[written:]


def _publish_immutable(
    path: Path,
    document: dict[str, Any],
    *,
    required_uid: int,
) -> Snapshot:
    """Publish one exact record with no replacement and a durable re-open.

    The staging name is never hard-linked.  If an exception is injected after
    the rename, the destination inode is checked before reporting failure so a
    caller can distinguish a committed publication from a pre-commit error.
    """

    if os.geteuid() != required_uid:
        fail("immutable publisher effective UID differs")
    raw = _canonical_json(document)
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        fail("immutable output exceeds its byte bound")
    path = _absolute(path, "immutable output", exists=False)
    parent_fd = _open_parent(path, "immutable output", required_uid=required_uid)
    temporary = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}"
    descriptor = -1
    renamed = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o444
            or staged.st_nlink != 1
            or staged.st_uid != required_uid
            or staged.st_size != len(raw)
        ):
            fail("immutable staged output metadata differs")
        try:
            _rename_noreplace(parent_fd, temporary, path.name)
            renamed = True
        except BaseException:
            try:
                published = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                renamed = published.st_dev == staged.st_dev and published.st_ino == staged.st_ino
            except OSError:
                renamed = False
            if not renamed:
                raise
        os.fsync(parent_fd)
        committed = _committed_immutable(path, raw, required_uid=required_uid)
        if committed is None:
            raise ImmutablePublicationCommittedError(
                "immutable output was renamed but durable validation failed"
            )
        return committed
    except ImmutablePublicationCommittedError:
        raise
    except BaseException as error:
        if renamed:
            committed = _committed_immutable(path, raw, required_uid=required_uid)
            if committed is not None:
                raise ImmutablePublicationCommittedError(
                    "immutable output was committed before a later failure"
                ) from error
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                if renamed:
                    raise ImmutablePublicationCommittedError(
                        "immutable output was renamed but descriptor close failed"
                    ) from error
                raise
        if not renamed:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError as error:
            if renamed:
                raise ImmutablePublicationCommittedError(
                    "immutable output was renamed but parent close failed"
                ) from error
            raise


def _publish_or_load(
    path: Path,
    document: dict[str, Any],
    *,
    required_uid: int,
) -> Snapshot:
    raw = _canonical_json(document)
    existing = _committed_immutable(path, raw, required_uid=required_uid)
    if existing is not None:
        return existing
    try:
        return _publish_immutable(path, document, required_uid=required_uid)
    except ImmutablePublicationCommittedError:
        committed = _committed_immutable(path, raw, required_uid=required_uid)
        if committed is not None:
            return committed
        raise


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} is invalid")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        fail(f"{label} is unsafe")
    return path


def _tree_seal(path: Path, label: str, *, required_uid: int) -> dict[str, Any]:
    """Seal a protected data tree by stable metadata inventory, not a copy."""

    path = _absolute(path, label, exists=True)
    root = path.lstat()
    if (
        stat.S_ISLNK(root.st_mode)
        or not stat.S_ISDIR(root.st_mode)
        or root.st_uid != required_uid
        or stat.S_IMODE(root.st_mode) != 0o555
    ):
        fail(f"{label} root seal differs")
    entries: list[bytes] = []
    regular_files = 0
    total_bytes = 0
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        metadata = current_path.lstat()
        relative = current_path.relative_to(path).as_posix() if current_path != path else "."
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) != 0o555
        ):
            fail(f"{label} directory seal differs: {relative}")
        entries.append(
            f"d\0{relative}\0{stat.S_IMODE(metadata.st_mode):o}\0{metadata.st_uid}\0{metadata.st_gid}\0{metadata.st_ino}\0{metadata.st_mtime_ns}\n".encode("ascii")
        )
        for name in directories + files:
            child = current_path / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode) or not (
                stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISREG(child_metadata.st_mode)
            ):
                fail(f"{label} contains a link or special file")
        for name in files:
            child = current_path / name
            metadata = child.lstat()
            relative = child.relative_to(path).as_posix()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != required_uid
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                fail(f"{label} regular-file seal differs: {relative}")
            regular_files += 1
            total_bytes += metadata.st_size
            entries.append(
                f"f\0{relative}\0{metadata.st_size}\0{metadata.st_ino}\0{metadata.st_mtime_ns}\0{metadata.st_ctime_ns}\n".encode("utf-8")
            )
    inventory = hashlib.sha256(b"".join(sorted(entries))).hexdigest()
    return {
        "path": os.fspath(path),
        "regular_files": regular_files,
        "total_bytes": total_bytes,
        "inventory_sha256": inventory,
    }


def _verify_tree_seal(seal: dict[str, Any], label: str, *, required_uid: int) -> None:
    _exact(seal, {"path", "regular_files", "total_bytes", "inventory_sha256"}, label)
    observed = _tree_seal(
        _absolute(seal["path"], f"{label}.path", exists=True),
        label,
        required_uid=required_uid,
    )
    if observed != seal:
        fail(f"{label} seal drifted")


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", os.fspath(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ActivationError("Git source inspection failed") from error
    if completed.returncode != 0:
        fail("Git source inspection failed")
    return completed.stdout.strip()


def _git_optional_output(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", os.fspath(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ActivationError("Git source inspection failed") from error
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _safe_source_root(path: Path, label: str, *, required_uid: int) -> Path:
    root = _absolute(path, label, exists=True)
    metadata = root.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != required_uid
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        fail(f"{label} root seal differs")
    return root


def _capture_git_source(
    root: Path,
    label: str,
    *,
    expected_commit: str,
    required_uid: int,
    tool_paths: Iterable[Path] = (),
    require_route_tools: bool = False,
) -> dict[str, Any]:
    """Bind a clean detached standalone clone and the exact tool bytes used."""

    root = _safe_source_root(root, label, required_uid=required_uid)
    expected_commit = _commit(expected_commit, f"{label}.expected_commit")
    try:
        canonical = root.resolve(strict=True)
    except OSError as error:
        raise ActivationError(f"{label} root is unavailable") from error
    if canonical != root:
        fail(f"{label} root is not canonical")
    git_dir = root / ".git"
    git_metadata = git_dir.lstat() if git_dir.exists() or git_dir.is_symlink() else None
    if (
        git_metadata is None
        or stat.S_ISLNK(git_metadata.st_mode)
        or not stat.S_ISDIR(git_metadata.st_mode)
        or git_metadata.st_uid != required_uid
        or stat.S_IMODE(git_metadata.st_mode) & 0o022
    ):
        fail(f"{label} is not a standalone Git clone")
    if _git_output(root, "rev-parse", "--show-toplevel") != os.fspath(root):
        fail(f"{label} Git root differs")
    if _git_output(root, "rev-parse", "--is-inside-work-tree") != "true":
        fail(f"{label} is not a work tree")
    if _git_output(root, "rev-parse", "--is-shallow-repository") != "false":
        fail(f"{label} is shallow")
    common = _git_output(root, "rev-parse", "--git-common-dir")
    git_directory = _git_output(root, "rev-parse", "--git-dir")
    if common != ".git" or git_directory != ".git":
        fail(f"{label} is a linked worktree")
    alternates = git_dir / "objects/info/alternates"
    if alternates.exists() or alternates.is_symlink():
        fail(f"{label} uses Git alternates")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        fail(f"{label} is dirty")
    if _git_optional_output(root, "symbolic-ref", "-q", "HEAD"):
        fail(f"{label} is not detached")
    commit = _git_output(root, "rev-parse", "HEAD")
    tree = _git_output(root, "rev-parse", "HEAD^{tree}")
    if commit != expected_commit or COMMIT_RE.fullmatch(tree) is None:
        fail(f"{label} commit or tree differs")
    requested_tools = {Path(item) for item in tool_paths}
    if require_route_tools:
        expected_tools = {root / relative for relative in CONTROL_TOOL_RELATIVE_PATHS}
        if requested_tools != expected_tools:
            fail("activation control source must seal every dedicated route tool")
    tools: list[dict[str, Any]] = []
    for tool in sorted(requested_tools, key=os.fspath):
        tool = _absolute(tool, f"{label} tool", exists=True)
        try:
            tool.relative_to(root)
        except ValueError as error:
            raise ActivationError(f"{label} tool is outside source root") from error
        snapshot = _snapshot(
            tool,
            f"{label} tool",
            maximum=MAX_DOCUMENT_BYTES,
            required_uid=required_uid,
            immutable=True,
        )
        tools.append({"path": os.fspath(tool), "sha256": snapshot.sha256})
    return {
        "root": os.fspath(root),
        "commit": commit,
        "tree": tree,
        "tools": tools,
    }


def _verify_git_source(
    seal: dict[str, Any],
    label: str,
    *,
    required_uid: int,
) -> None:
    _exact(seal, {"root", "commit", "tree", "tools"}, label)
    tools_value = seal["tools"]
    if not isinstance(tools_value, list):
        fail(f"{label}.tools differs")
    tools: list[Path] = []
    for item in tools_value:
        _exact(item, {"path", "sha256"}, f"{label}.tool")
        _hash(item["sha256"], f"{label}.tool.sha256")
        tools.append(_absolute(item["path"], f"{label}.tool.path", exists=True))
    observed = _capture_git_source(
        _absolute(seal["root"], f"{label}.root", exists=True),
        label,
        expected_commit=_commit(seal["commit"], f"{label}.commit"),
        required_uid=required_uid,
        tool_paths=tools,
        require_route_tools=(label == "activation control source"),
    )
    if observed != seal:
        fail(f"{label} seal drifted")


def _manifest_identity(raw: bytes, label: str) -> dict[str, Any]:
    document = _strict_object(raw, label)
    if document.get("schema_version") != SERVED_MODEL_SCHEMA:
        fail(f"{label} schema differs")
    public = document.get("public")
    format_value = document.get("format")
    worker = document.get("worker")
    product = document.get("product")
    tokenizer = document.get("tokenizer")
    promotion = document.get("promotion")
    if not all(
        isinstance(value, dict)
        for value in (public, format_value, worker, product, tokenizer, promotion)
    ):
        fail(f"{label} required object differs")
    model_id = public.get("id")
    format_id = format_value.get("format_id")
    protocol = worker.get("protocol")
    worker_path = worker.get("binary")
    worker_sha = worker.get("binary_sha256")
    product_root = product.get("root")
    package = product.get("package")
    tokenizer_root = tokenizer.get("root")
    tokenizer_files = tokenizer.get("files")
    source_commit = promotion.get("source_commit")
    receipt = promotion.get("receipt")
    receipt_sha = promotion.get("receipt_sha256")
    if (
        not isinstance(model_id, str)
        or not isinstance(format_id, str)
        or not isinstance(protocol, str)
        or not isinstance(package, dict)
        or not isinstance(tokenizer_files, dict)
    ):
        fail(f"{label} identity differs")
    package_path = _safe_relative(package.get("manifest_path"), f"{label}.package.manifest_path")
    package_sha = _hash(package.get("manifest_sha256"), f"{label}.package.manifest_sha256")
    if not tokenizer_files or not all(
        isinstance(key, str) and _safe_relative(key, f"{label}.tokenizer.files")
        and isinstance(value, str)
        and HASH_RE.fullmatch(value)
        for key, value in tokenizer_files.items()
    ):
        fail(f"{label}.tokenizer.files differs")
    return {
        "model_id": model_id,
        "format_id": format_id,
        "worker_protocol": protocol,
        "worker_path": os.fspath(_absolute(worker_path, f"{label}.worker.binary", exists=True)),
        "worker_sha256": _hash(worker_sha, f"{label}.worker.binary_sha256"),
        "product_root": os.fspath(_absolute(product_root, f"{label}.product.root", exists=True)),
        "package_manifest_path": package_path.as_posix(),
        "package_manifest_sha256": package_sha,
        "tokenizer_root": os.fspath(_absolute(tokenizer_root, f"{label}.tokenizer.root", exists=True)),
        "tokenizer_files": dict(sorted(tokenizer_files.items())),
        "promotion_source_commit": _commit(source_commit, f"{label}.promotion.source_commit"),
        "promotion_receipt_path": os.fspath(
            _absolute(receipt, f"{label}.promotion.receipt", exists=True)
        ),
        "promotion_receipt_sha256": _hash(
            receipt_sha, f"{label}.promotion.receipt_sha256"
        ),
    }


def _manifest_snapshot(
    path: Path,
    label: str,
    *,
    required_uid: int,
    immutable: bool,
) -> tuple[Snapshot, dict[str, Any]]:
    snapshot = _snapshot(
        path,
        label,
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=immutable,
    )
    return snapshot, _manifest_identity(snapshot.raw, label)


def _tokenizer_seals(
    identity: dict[str, Any],
    label: str,
    *,
    required_uid: int,
) -> list[dict[str, Any]]:
    root = Path(identity["tokenizer_root"])
    result: list[dict[str, Any]] = []
    files = identity["tokenizer_files"]
    assert isinstance(files, dict)
    for relative, expected_sha in files.items():
        path = root / _safe_relative(relative, f"{label}.tokenizer file")
        snapshot = _snapshot(
            path,
            f"{label}.tokenizer file",
            required_uid=required_uid,
            immutable=True,
        )
        if snapshot.sha256 != expected_sha:
            fail(f"{label}.tokenizer file SHA-256 differs")
        result.append(_seal(snapshot))
    return result


def _capture_protected_runtime(
    manifest: Snapshot,
    identity: dict[str, Any],
    promotion_source: Path,
    *,
    required_uid: int,
) -> dict[str, Any]:
    if identity["format_id"] != AQ4_FORMAT_ID or identity["worker_protocol"] != WORKER_PROTOCOL:
        fail("candidate manifest is not AQ4_0/worker-v2")
    worker = _snapshot(
        Path(identity["worker_path"]),
        "candidate worker",
        required_uid=required_uid,
        immutable=False,
        executable=True,
    )
    if worker.sha256 != identity["worker_sha256"]:
        fail("candidate worker bytes differ from manifest")
    package_manifest = _snapshot(
        Path(identity["product_root"])
        / _safe_relative(identity["package_manifest_path"], "candidate package manifest"),
        "candidate package manifest",
        required_uid=required_uid,
        immutable=True,
    )
    if package_manifest.sha256 != identity["package_manifest_sha256"]:
        fail("candidate package manifest bytes differ")
    receipt = _snapshot(
        Path(identity["promotion_receipt_path"]),
        "candidate promotion receipt",
        required_uid=required_uid,
        immutable=True,
    )
    if receipt.sha256 != identity["promotion_receipt_sha256"]:
        fail("candidate promotion receipt bytes differ")
    return {
        "manifest": _seal(manifest),
        "worker": _seal(worker),
        "product_tree": _tree_seal(
            Path(identity["product_root"]), "candidate product tree", required_uid=required_uid
        ),
        "package_manifest": _seal(package_manifest),
        "tokenizer_tree": _tree_seal(
            Path(identity["tokenizer_root"]), "candidate tokenizer tree", required_uid=required_uid
        ),
        "tokenizer_files": _tokenizer_seals(identity, "candidate", required_uid=required_uid),
        "promotion_receipt": _seal(receipt),
        "promotion_source": _capture_git_source(
            promotion_source,
            "AQ4 promotion source",
            expected_commit=identity["promotion_source_commit"],
            required_uid=required_uid,
        ),
    }


def _verify_protected_runtime(runtime: dict[str, Any], *, required_uid: int) -> None:
    fields = {
        "manifest",
        "worker",
        "product_tree",
        "package_manifest",
        "tokenizer_tree",
        "tokenizer_files",
        "promotion_receipt",
        "promotion_source",
    }
    _exact(runtime, fields, "candidate runtime")
    manifest = _verify_seal(
        runtime["manifest"], "candidate frozen manifest", required_uid=required_uid, immutable=True
    )
    identity = _manifest_identity(manifest.raw, "candidate frozen manifest")
    worker = _verify_seal(
        runtime["worker"], "candidate worker", required_uid=required_uid, immutable=False, executable=True
    )
    if worker.sha256 != identity["worker_sha256"]:
        fail("candidate worker no longer matches manifest")
    _verify_tree_seal(runtime["product_tree"], "candidate product tree", required_uid=required_uid)
    _verify_seal(
        runtime["package_manifest"], "candidate package manifest", required_uid=required_uid, immutable=True
    )
    _verify_tree_seal(runtime["tokenizer_tree"], "candidate tokenizer tree", required_uid=required_uid)
    tokenizer_files = runtime["tokenizer_files"]
    if not isinstance(tokenizer_files, list):
        fail("candidate tokenizer seals differ")
    expected_by_path = {
        os.fspath(Path(identity["tokenizer_root"]) / relative): digest
        for relative, digest in identity["tokenizer_files"].items()
    }
    if len(tokenizer_files) != len(expected_by_path):
        fail("candidate tokenizer seal count differs")
    for seal in tokenizer_files:
        snapshot = _verify_seal(
            seal, "candidate tokenizer file", required_uid=required_uid, immutable=True
        )
        if expected_by_path.get(os.fspath(snapshot.path)) != snapshot.sha256:
            fail("candidate tokenizer seal differs from manifest")
    _verify_seal(
        runtime["promotion_receipt"], "candidate promotion receipt", required_uid=required_uid, immutable=True
    )
    _verify_git_source(runtime["promotion_source"], "AQ4 promotion source", required_uid=required_uid)


def _capture_legacy_runtime(
    rollback: Snapshot,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Capture only hash-bearing legacy inputs; old AQ4 is intentionally unsealed."""

    worker_path = Path(identity["worker_path"])
    if worker_path.is_symlink() or not worker_path.is_file():
        fail("legacy worker is unavailable")
    worker_raw = worker_path.read_bytes()
    if _sha256(worker_raw) != identity["worker_sha256"]:
        fail("legacy worker SHA-256 differs")
    package = Path(identity["product_root"]) / identity["package_manifest_path"]
    if package.is_symlink() or not package.is_file():
        fail("legacy package manifest is unavailable")
    package_raw = package.read_bytes()
    if _sha256(package_raw) != identity["package_manifest_sha256"]:
        fail("legacy package manifest SHA-256 differs")
    receipt = Path(identity["promotion_receipt_path"])
    if receipt.is_symlink() or not receipt.is_file():
        fail("legacy promotion receipt is unavailable")
    receipt_raw = receipt.read_bytes()
    if _sha256(receipt_raw) != identity["promotion_receipt_sha256"]:
        fail("legacy promotion receipt SHA-256 differs")
    tokenizer_hashes: dict[str, str] = {}
    for relative, expected in identity["tokenizer_files"].items():
        item = Path(identity["tokenizer_root"]) / relative
        if item.is_symlink() or not item.is_file():
            fail("legacy tokenizer file is unavailable")
        observed = _sha256(item.read_bytes())
        if observed != expected:
            fail("legacy tokenizer file SHA-256 differs")
        tokenizer_hashes[relative] = observed
    return {
        "rollback_manifest_sha256": rollback.sha256,
        "worker_path": os.fspath(worker_path),
        "worker_sha256": _sha256(worker_raw),
        "package_manifest_path": os.fspath(package),
        "package_manifest_sha256": _sha256(package_raw),
        "promotion_receipt_path": os.fspath(receipt),
        "promotion_receipt_sha256": _sha256(receipt_raw),
        "tokenizer_files": tokenizer_hashes,
    }


def _verify_legacy_runtime(seal: dict[str, Any], rollback: Snapshot) -> None:
    fields = {
        "rollback_manifest_sha256",
        "worker_path",
        "worker_sha256",
        "package_manifest_path",
        "package_manifest_sha256",
        "promotion_receipt_path",
        "promotion_receipt_sha256",
        "tokenizer_files",
    }
    _exact(seal, fields, "legacy runtime")
    identity = _manifest_identity(rollback.raw, "rollback manifest")
    observed = _capture_legacy_runtime(rollback, identity)
    if observed != seal:
        fail("legacy runtime asset/hash seal drifted")


def _capture_operations(
    path: Path,
    *,
    required_uid: int,
) -> dict[str, Any]:
    snapshot = _snapshot(
        path,
        "reviewed AQ4 activation operations",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    document = _strict_object(snapshot.raw, "reviewed AQ4 activation operations")
    _exact(document, {"schema_version", "stages"}, "reviewed AQ4 activation operations")
    if document["schema_version"] != OPERATIONS_SCHEMA:
        fail("reviewed AQ4 activation operations schema differs")
    stages = _exact(
        document["stages"],
        set(OPERATION_STAGES),
        "reviewed AQ4 activation operation stages",
    )
    captured: dict[str, dict[str, Any]] = {}
    for stage in OPERATION_STAGES:
        operation = _exact(
            stages[stage],
            {"argv", "executable_sha256", "timeout_seconds"},
            f"reviewed operation {stage}",
        )
        argv = operation["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(value, str) and value for value in argv)
            or not Path(argv[0]).is_absolute()
        ):
            fail(f"reviewed operation {stage} argv differs")
        executable = _snapshot(
            Path(argv[0]),
            f"reviewed operation {stage} executable",
            required_uid=required_uid,
            immutable=False,
            executable=True,
        )
        if executable.sha256 != _hash(
            operation["executable_sha256"], f"reviewed operation {stage} executable SHA-256"
        ):
            fail(f"reviewed operation {stage} executable SHA-256 differs")
        timeout = operation["timeout_seconds"]
        if type(timeout) is not int or not 1 <= timeout <= 600:
            fail(f"reviewed operation {stage} timeout differs")
        captured[stage] = {
            "argv": argv,
            "timeout_seconds": timeout,
            "executable": _seal(executable),
        }
    return {
        "path": os.fspath(snapshot.path),
        "sha256": snapshot.sha256,
        "stages": captured,
    }


def _verify_operations(seal: dict[str, Any], *, required_uid: int) -> None:
    _exact(seal, {"path", "sha256", "stages"}, "reviewed operations seal")
    observed = _capture_operations(
        _absolute(seal["path"], "reviewed operations path", exists=True),
        required_uid=required_uid,
    )
    if observed != seal:
        fail("reviewed operations seal drifted")


def _capture_preconditions(
    systemd_unit: Path,
    environment_file: Path,
    credential_files: Sequence[Path],
    *,
    service_unit: str,
    required_uid: int,
) -> dict[str, Any]:
    if not isinstance(service_unit, str) or service_unit != DEFAULT_SERVICE_UNIT:
        fail("service unit must be ullm-openai.service")
    if not credential_files:
        fail("at least one credential seal is required")
    unit = _snapshot(
        systemd_unit,
        "systemd unit",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=False,
    )
    environment = _snapshot(
        environment_file,
        "gateway environment file",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=False,
    )
    credentials = [
        _seal(
            _snapshot(
                path,
                "gateway credential",
                maximum=MAX_DOCUMENT_BYTES,
                required_uid=required_uid,
                immutable=False,
            )
        )
        for path in sorted({Path(item) for item in credential_files}, key=os.fspath)
    ]
    return {
        "service_unit": service_unit,
        "systemd_unit": _seal(unit),
        "environment_file": _seal(environment),
        "credentials": credentials,
    }


def _verify_preconditions(preconditions: dict[str, Any], *, required_uid: int) -> None:
    _exact(
        preconditions,
        {"service_unit", "systemd_unit", "environment_file", "credentials"},
        "runtime preconditions",
    )
    if preconditions["service_unit"] != DEFAULT_SERVICE_UNIT:
        fail("runtime precondition service differs")
    _verify_seal(
        preconditions["systemd_unit"], "systemd unit", required_uid=required_uid, immutable=False
    )
    _verify_seal(
        preconditions["environment_file"],
        "gateway environment file",
        required_uid=required_uid,
        immutable=False,
    )
    credentials = preconditions["credentials"]
    if not isinstance(credentials, list) or not credentials:
        fail("credential seals differ")
    for seal in credentials:
        _verify_seal(seal, "gateway credential", required_uid=required_uid, immutable=False)


def _require_below(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ActivationError(f"{label} is outside the protected AQ4 root") from error


def _validate_output_path(path: Path, label: str, *, required_uid: int) -> Path:
    path = _absolute(path, label, exists=False)
    if not path.name or path.name in {".", ".."}:
        fail(f"{label} basename differs")
    descriptor = _open_parent(path, label, required_uid=required_uid)
    os.close(descriptor)
    return path


def _validate_audit_directory(path: Path, label: str, *, required_uid: int) -> Path:
    path = _absolute(path, label, exists=True)
    descriptor = -1
    try:
        descriptor = _walk_directory(path)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            fail(f"{label} is unsafe")
        return path
    except OSError as error:
        raise ActivationError(f"{label} is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _lock_metadata(path: Path, *, required_uid: int) -> dict[str, Any]:
    snapshot = _snapshot(
        path,
        "AQ4 activation lock",
        maximum=1024,
        required_uid=required_uid,
        immutable=False,
        allow_empty=True,
    )
    if stat.S_IMODE(snapshot.identity.mode) != 0o600:
        fail("AQ4 activation lock must be mode 0600")
    return _seal(snapshot)


def _manifest_compatibility(
    candidate: dict[str, Any],
    rollback: dict[str, Any],
    *,
    expected_model_id: str,
    expected_worker_sha256: str,
) -> None:
    _hash(expected_worker_sha256, "expected AQ4 worker SHA-256")
    if (
        candidate["model_id"] != expected_model_id
        or rollback["model_id"] != expected_model_id
        or candidate["format_id"] != AQ4_FORMAT_ID
        or rollback["format_id"] != AQ4_FORMAT_ID
        or candidate["worker_protocol"] != WORKER_PROTOCOL
        or rollback["worker_protocol"] != WORKER_PROTOCOL
        or candidate["worker_sha256"] != expected_worker_sha256
        or rollback["worker_sha256"] != expected_worker_sha256
    ):
        fail("AQ4-to-AQ4 manifest compatibility differs")
    if (
        candidate["worker_path"] == rollback["worker_path"]
        or candidate["product_root"] == rollback["product_root"]
        or candidate["tokenizer_root"] == rollback["tokenizer_root"]
    ):
        fail("candidate does not move all AQ4 runtime closure paths")


def _output_document_paths(outputs: dict[str, Any]) -> list[Path]:
    names = {
        "activation_intent_path",
        "activation_outcome_path",
        "activation_recovery_path",
        "rollback_outcome_path",
        "candidate_live_proof_path",
        "rollback_live_proof_path",
        "candidate_isolated_preflight_path",
    }
    _exact(
        outputs,
        names
        | {
            "recovery_audit_directory",
            "rollback_audit_directory",
            "live_proof_audit_directory",
        },
        "activation output destinations",
    )
    return [Path(outputs[name]) for name in sorted(names)]


def prepare_plan(
    *,
    plan_id: str,
    protected_root: Path,
    control_source: Path,
    control_source_commit: str,
    control_tool_paths: Sequence[Path],
    promotion_source: Path,
    candidate_manifest: Path,
    active_manifest: Path,
    rollback_manifest: Path,
    systemd_unit: Path,
    environment_file: Path,
    credential_files: Sequence[Path],
    operations_document: Path,
    lock_path: Path,
    activation_intent: Path,
    activation_outcome: Path,
    activation_recovery: Path,
    rollback_outcome: Path,
    candidate_live_proof: Path,
    rollback_live_proof: Path,
    candidate_isolated_preflight: Path,
    recovery_audit_directory: Path,
    rollback_audit_directory: Path,
    live_proof_audit_directory: Path,
    output: Path,
    expected_model_id: str = AQ4_MODEL_ID,
    expected_worker_sha256: str = EXPECTED_AQ4_WORKER_SHA256,
    service_unit: str = DEFAULT_SERVICE_UNIT,
    required_uid: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prepare the one immutable AQ4 hardening plan; this never swaps active bytes."""

    if os.geteuid() != required_uid:
        fail("plan preparation effective UID differs")
    timestamp = utc_now() if now is None else now
    plan_id = _identifier(plan_id, "plan_id")
    protected_root = _safe_source_root(
        protected_root, "protected AQ4 root", required_uid=required_uid
    )
    candidate_snapshot, candidate_identity = _manifest_snapshot(
        candidate_manifest, "candidate frozen manifest", required_uid=required_uid, immutable=True
    )
    rollback_snapshot, rollback_identity = _manifest_snapshot(
        rollback_manifest, "rollback manifest", required_uid=required_uid, immutable=True
    )
    active_snapshot, _active_identity = _manifest_snapshot(
        active_manifest, "active manifest", required_uid=required_uid, immutable=False
    )
    if active_snapshot.raw != rollback_snapshot.raw:
        fail("active manifest bytes do not equal exact saved rollback bytes")
    if candidate_snapshot.raw == rollback_snapshot.raw:
        fail("candidate manifest bytes equal rollback bytes")
    _manifest_compatibility(
        candidate_identity,
        rollback_identity,
        expected_model_id=expected_model_id,
        expected_worker_sha256=expected_worker_sha256,
    )
    for path, label in (
        (candidate_snapshot.path, "candidate manifest"),
        (Path(candidate_identity["worker_path"]), "candidate worker"),
        (Path(candidate_identity["product_root"]), "candidate product"),
        (Path(candidate_identity["tokenizer_root"]), "candidate tokenizer"),
        (promotion_source, "AQ4 promotion source"),
        (control_source, "activation control source"),
    ):
        _require_below(_absolute(path, label, exists=True), protected_root, label)
    control = _capture_git_source(
        control_source,
        "activation control source",
        expected_commit=control_source_commit,
        required_uid=required_uid,
        tool_paths=control_tool_paths,
        require_route_tools=True,
    )
    candidate_runtime = _capture_protected_runtime(
        candidate_snapshot,
        candidate_identity,
        promotion_source,
        required_uid=required_uid,
    )
    legacy_runtime = _capture_legacy_runtime(rollback_snapshot, rollback_identity)
    preconditions = _capture_preconditions(
        systemd_unit,
        environment_file,
        credential_files,
        service_unit=service_unit,
        required_uid=required_uid,
    )
    operations = _capture_operations(operations_document, required_uid=required_uid)
    lock = _lock_metadata(lock_path, required_uid=required_uid)
    output_values = {
        "activation_intent_path": activation_intent,
        "activation_outcome_path": activation_outcome,
        "activation_recovery_path": activation_recovery,
        "rollback_outcome_path": rollback_outcome,
        "candidate_live_proof_path": candidate_live_proof,
        "rollback_live_proof_path": rollback_live_proof,
        "candidate_isolated_preflight_path": candidate_isolated_preflight,
        "recovery_audit_directory": recovery_audit_directory,
        "rollback_audit_directory": rollback_audit_directory,
        "live_proof_audit_directory": live_proof_audit_directory,
    }
    outputs = {
        name: os.fspath(_absolute(value, name, exists=None))
        for name, value in output_values.items()
    }
    output_paths = _output_document_paths(outputs)
    if len({os.fspath(path) for path in output_paths}) != len(output_paths):
        fail("activation output paths collide")
    for path in output_paths:
        _require_below(path, protected_root, "activation output")
        _validate_output_path(path, "activation output", required_uid=required_uid)
    for name in (
        "recovery_audit_directory",
        "rollback_audit_directory",
        "live_proof_audit_directory",
    ):
        directory = _validate_audit_directory(
            Path(outputs[name]), name, required_uid=required_uid
        )
        _require_below(directory, protected_root, name)
    output = _validate_output_path(output, "activation plan output", required_uid=required_uid)
    _require_below(output, protected_root, "activation plan output")
    if output in output_paths:
        fail("activation plan output collides with an outcome")
    document = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": plan_id,
        "created_at": utc_timestamp(timestamp),
        "operation_epoch": secrets.token_hex(32),
        "protected_root": os.fspath(protected_root),
        "expected": {
            "model_id": expected_model_id,
            "format_id": AQ4_FORMAT_ID,
            "worker_protocol": WORKER_PROTOCOL,
            "worker_binary_sha256": expected_worker_sha256,
        },
        "control_source": control,
        "candidate_runtime": candidate_runtime,
        "rollback_manifest": _seal(rollback_snapshot),
        "active_manifest": {
            "path": os.fspath(active_snapshot.path),
            "sha256": active_snapshot.sha256,
            "bytes": active_snapshot.identity.size,
        },
        "legacy_runtime": legacy_runtime,
        "runtime_preconditions": preconditions,
        "operations": operations,
        "lock": lock,
        "outcomes": outputs,
    }
    _publish_immutable(output, document, required_uid=required_uid)
    return document


def _validate_plan_document(document: dict[str, Any]) -> None:
    _exact(
        document,
        {
            "schema_version",
            "plan_id",
            "created_at",
            "operation_epoch",
            "protected_root",
            "expected",
            "control_source",
            "candidate_runtime",
            "rollback_manifest",
            "active_manifest",
            "legacy_runtime",
            "runtime_preconditions",
            "operations",
            "lock",
            "outcomes",
        },
        "AQ4 hardening activation plan",
    )
    if document["schema_version"] != PLAN_SCHEMA:
        fail("AQ4 hardening activation plan schema differs")
    _identifier(document["plan_id"], "AQ4 hardening activation plan.plan_id")
    _hash(document["operation_epoch"], "AQ4 hardening activation plan.operation_epoch")
    protected = _absolute(
        document["protected_root"], "AQ4 hardening activation plan.protected_root", exists=True
    )
    metadata = protected.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail("AQ4 hardening activation protected root differs")
    expected = _exact(
        document["expected"],
        {"model_id", "format_id", "worker_protocol", "worker_binary_sha256"},
        "AQ4 hardening activation plan expected identity",
    )
    if (
        expected["format_id"] != AQ4_FORMAT_ID
        or expected["worker_protocol"] != WORKER_PROTOCOL
        or not isinstance(expected["model_id"], str)
    ):
        fail("AQ4 hardening activation expected identity differs")
    _hash(expected["worker_binary_sha256"], "AQ4 expected worker SHA-256")
    _exact(
        document["active_manifest"], {"path", "sha256", "bytes"}, "active manifest binding"
    )
    _absolute(document["active_manifest"]["path"], "active manifest binding.path")
    _hash(document["active_manifest"]["sha256"], "active manifest binding.sha256")
    if type(document["active_manifest"]["bytes"]) is not int or document["active_manifest"]["bytes"] < 1:
        fail("active manifest binding.bytes differs")
    _output_document_paths(document["outcomes"])


def load_plan(path: Path, *, required_uid: int = 0) -> PlanRecord:
    snapshot = _snapshot(
        path,
        "AQ4 hardening activation plan",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    document = _strict_object(snapshot.raw, "AQ4 hardening activation plan")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 hardening activation plan is not canonical JSON")
    _validate_plan_document(document)
    return PlanRecord(snapshot, document)


def _verify_plan_inputs(
    record: PlanRecord,
    *,
    required_uid: int,
    include_candidate: bool = True,
    include_legacy: bool = True,
) -> tuple[Snapshot, Snapshot, Snapshot]:
    """Re-pin every mutation-relevant input before the locked boundary."""

    document = record.document
    protected_root = _safe_source_root(
        Path(document["protected_root"]), "protected AQ4 root", required_uid=required_uid
    )
    _verify_git_source(
        document["control_source"], "activation control source", required_uid=required_uid
    )
    _verify_operations(document["operations"], required_uid=required_uid)
    _verify_preconditions(document["runtime_preconditions"], required_uid=required_uid)
    lock = _lock_metadata(Path(document["lock"]["path"]), required_uid=required_uid)
    if lock != document["lock"]:
        fail("AQ4 activation lock seal drifted")
    rollback = _verify_seal(
        document["rollback_manifest"],
        "saved rollback manifest",
        required_uid=required_uid,
        immutable=True,
    )
    rollback_identity = _manifest_identity(rollback.raw, "saved rollback manifest")
    active_binding = document["active_manifest"]
    active = _snapshot(
        Path(active_binding["path"]),
        "active manifest",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=False,
    )
    if (
        active.sha256 != active_binding["sha256"]
        or active.identity.size != active_binding["bytes"]
        or active.raw != rollback.raw
    ):
        fail("active manifest differs from the exact rollback bytes")
    candidate = _verify_seal(
        document["candidate_runtime"]["manifest"],
        "candidate frozen manifest",
        required_uid=required_uid,
        immutable=True,
    )
    candidate_identity = _manifest_identity(candidate.raw, "candidate frozen manifest")
    _manifest_compatibility(
        candidate_identity,
        rollback_identity,
        expected_model_id=document["expected"]["model_id"],
        expected_worker_sha256=document["expected"]["worker_binary_sha256"],
    )
    for target, label in (
        (candidate.path, "candidate frozen manifest"),
        (Path(candidate_identity["worker_path"]), "candidate worker"),
        (Path(candidate_identity["product_root"]), "candidate product"),
        (Path(candidate_identity["tokenizer_root"]), "candidate tokenizer"),
        (Path(document["control_source"]["root"]), "activation control source"),
    ):
        _require_below(target, protected_root, label)
    if include_candidate:
        _verify_protected_runtime(document["candidate_runtime"], required_uid=required_uid)
    if include_legacy:
        _verify_legacy_runtime(document["legacy_runtime"], rollback)
    for path in _output_document_paths(document["outcomes"]):
        _require_below(path, protected_root, "activation output")
        _validate_output_path_or_committed(path, required_uid=required_uid)
    for name in (
        "recovery_audit_directory",
        "rollback_audit_directory",
        "live_proof_audit_directory",
    ):
        directory = _validate_audit_directory(
            Path(document["outcomes"][name]), name, required_uid=required_uid
        )
        _require_below(directory, protected_root, name)
    return active, rollback, candidate


def _validate_output_path_or_committed(path: Path, *, required_uid: int) -> None:
    """Only verify parent safety during read-only preflight; do not create files."""

    path = _absolute(path, "activation output")
    parent = _open_parent(path, "activation output", required_uid=required_uid)
    os.close(parent)
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
        ):
            fail("activation output exists with an unsafe identity")


def _output_unused(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        fail(f"{label} is already consumed")


def _preflight_checks(record: PlanRecord, *, required_uid: int) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    work: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "control_source",
            lambda: _verify_git_source(
                record.document["control_source"],
                "activation control source",
                required_uid=required_uid,
            ),
        ),
        (
            "candidate_runtime",
            lambda: _verify_protected_runtime(
                record.document["candidate_runtime"], required_uid=required_uid
            ),
        ),
        (
            "runtime_preconditions",
            lambda: _verify_preconditions(
                record.document["runtime_preconditions"], required_uid=required_uid
            ),
        ),
        (
            "reviewed_operations",
            lambda: _verify_operations(record.document["operations"], required_uid=required_uid),
        ),
        (
            "lock",
            lambda: _lock_metadata(
                Path(record.document["lock"]["path"]), required_uid=required_uid
            ),
        ),
        (
            "manifest_bindings",
            lambda: _verify_plan_inputs(
                record,
                required_uid=required_uid,
                include_candidate=False,
                include_legacy=True,
            ),
        ),
        (
            "candidate_isolated_preflight",
            lambda: _load_isolated_preflight(
                record,
                candidate=_verify_plan_inputs(
                    record,
                    required_uid=required_uid,
                    include_candidate=True,
                    include_legacy=False,
                )[2],
                required_uid=required_uid,
            ),
        ),
    )
    for name, action in work:
        try:
            action()
            checks[name] = {"passed": True}
        except Exception as error:
            checks[name] = {"passed": False, "reason": str(error)}
            failures.append(f"{name}: {error}")
    outputs = record.document["outcomes"]
    for name in (
        "activation_intent_path",
        "activation_outcome_path",
        "activation_recovery_path",
        "rollback_outcome_path",
    ):
        path = Path(outputs[name])
        if path.exists() or path.is_symlink():
            checks[f"unused_{name}"] = {"passed": False, "reason": "already consumed"}
            failures.append(f"{name}: already consumed")
        else:
            checks[f"unused_{name}"] = {"passed": True}
    return {"checks": checks, "failures": failures}


def preflight_report(record: PlanRecord, *, required_uid: int = 0) -> dict[str, Any]:
    """Read-only re-seal report.  It never creates a plan, lock, or outcome."""

    result = _preflight_checks(record, required_uid=required_uid)
    document = record.document
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "operation_epoch": document["operation_epoch"],
        "active_manifest_sha256": document["active_manifest"]["sha256"],
        "rollback_manifest_sha256": document["rollback_manifest"]["sha256"],
        "candidate_manifest_sha256": document["candidate_runtime"]["manifest"]["sha256"],
        "ready": not result["failures"],
        "checks": result["checks"],
        "blockers": result["failures"],
        "production_activation_performed": False,
    }


def _preplan_path_state(path: Path, *, required_uid: int) -> dict[str, Any]:
    path = _absolute(path, "pre-plan path")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"path": os.fspath(path), "exists": False}
    except OSError as error:
        return {"path": os.fspath(path), "exists": False, "error": type(error).__name__}
    if stat.S_ISLNK(metadata.st_mode):
        return {"path": os.fspath(path), "exists": True, "safe": False, "kind": "symlink"}
    kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "regular" if stat.S_ISREG(metadata.st_mode) else "special"
    return {
        "path": os.fspath(path),
        "exists": True,
        "safe": metadata.st_uid == required_uid and not (stat.S_IMODE(metadata.st_mode) & 0o022),
        "kind": kind,
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": metadata.st_uid,
        "nlink": metadata.st_nlink,
    }


def preplan_preflight_report(
    *,
    active_manifest: Path,
    expected_active_sha256: str,
    protected_root: Path,
    candidate_manifest: Path,
    rollback_manifest: Path,
    plan_path: Path,
    control_source_parent: Path,
    operations_document: Path,
    lock_path: Path,
    systemd_unit: Path,
    environment_file: Path,
    required_uid: int = 0,
) -> dict[str, Any]:
    """Read only the known Phase-5 admission locations before a plan exists.

    This is intentionally not a substitute for :func:`preflight_report`: a
    complete immutable plan is required before the route can ever report
    ``ready: true``.  It gives an operator concrete missing prerequisites
    without creating a lock, protected path, plan, or active-manifest write.
    """

    _hash(expected_active_sha256, "expected active manifest SHA-256")
    states = {
        "protected_root": _preplan_path_state(protected_root, required_uid=required_uid),
        "candidate_manifest": _preplan_path_state(candidate_manifest, required_uid=required_uid),
        "rollback_manifest": _preplan_path_state(rollback_manifest, required_uid=required_uid),
        "activation_plan": _preplan_path_state(plan_path, required_uid=required_uid),
        "control_source_parent": _preplan_path_state(control_source_parent, required_uid=required_uid),
        "reviewed_operations": _preplan_path_state(operations_document, required_uid=required_uid),
        "activation_lock": _preplan_path_state(lock_path, required_uid=required_uid),
    }
    blockers: list[str] = []
    for name, state in states.items():
        if state.get("exists") is not True:
            blockers.append(f"{name} is absent")
        elif state.get("safe") is not True:
            blockers.append(f"{name} is not root-owned/non-group-world-writable")
    observed: dict[str, Any] = {}
    try:
        active = _snapshot(
            active_manifest,
            "active manifest",
            maximum=MAX_DOCUMENT_BYTES,
            required_uid=required_uid,
            immutable=False,
        )
        observed["active_manifest"] = {
            "path": os.fspath(active.path),
            "sha256": active.sha256,
            "bytes": active.identity.size,
        }
        if active.sha256 != expected_active_sha256:
            blockers.append("active manifest SHA-256 differs from AQ4 hardening admission value")
    except Exception as error:
        observed["active_manifest"] = {"error": type(error).__name__}
        blockers.append("active manifest cannot be sealed for pre-plan admission")
    for name, path in (("systemd_unit", systemd_unit), ("environment_file", environment_file)):
        try:
            snapshot = _snapshot(
                path,
                name,
                maximum=MAX_DOCUMENT_BYTES,
                required_uid=required_uid,
                immutable=False,
            )
            observed[name] = {"path": os.fspath(snapshot.path), "sha256": snapshot.sha256}
        except Exception as error:
            observed[name] = {"error": type(error).__name__}
            blockers.append(f"{name} cannot be sealed for pre-plan admission")
    blockers.append("credential seal set is not configured until the reviewed activation plan is prepared")
    return {
        "schema_version": PREPLAN_PREFLIGHT_SCHEMA,
        "ready": False,
        "mode": "read_only_preplan",
        "expected_active_manifest_sha256": expected_active_sha256,
        "observed": observed,
        "paths": states,
        "blockers": blockers,
        "production_activation_performed": False,
    }


def _open_locked_activation_lock(record: PlanRecord, *, required_uid: int) -> int:
    seal = record.document["lock"]
    _exact(
        seal,
        {
            "path",
            "sha256",
            "bytes",
            "mode",
            "uid",
            "gid",
            "nlink",
            "device",
            "inode",
            "mtime_ns",
            "ctime_ns",
        },
        "AQ4 activation lock",
    )
    path = _absolute(seal["path"], "AQ4 activation lock", exists=True)
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        current = _lock_metadata(path, required_uid=required_uid)
        if current != seal:
            fail("AQ4 activation lock seal drifted")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ActivationError("AQ4 activation lock is already held") from error
        current = _lock_metadata(path, required_uid=required_uid)
        if current != seal:
            fail("AQ4 activation lock changed while being acquired")
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _require_same_plan(initial: PlanRecord, locked: PlanRecord, expected_sha256: str) -> None:
    if (
        expected_sha256 != initial.snapshot.sha256
        or expected_sha256 != locked.snapshot.sha256
        or initial.snapshot.path != locked.snapshot.path
        or initial.snapshot.raw != locked.snapshot.raw
        or initial.snapshot.identity != locked.snapshot.identity
    ):
        fail("confirmed AQ4 hardening plan changed before the locked boundary")


def _intent_document(record: PlanRecord, *, created_at: datetime) -> dict[str, Any]:
    active = Path(record.document["active_manifest"]["path"])
    staging_name = f".{active.name}.aq4-hardening-{record.snapshot.sha256[:24]}.exchange"
    return {
        "schema_version": INTENT_SCHEMA,
        "plan_id": record.document["plan_id"],
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "created_at": utc_timestamp(created_at),
        "active_manifest_path": os.fspath(active),
        "rollback_manifest_sha256": record.document["rollback_manifest"]["sha256"],
        "candidate_manifest_sha256": record.document["candidate_runtime"]["manifest"]["sha256"],
        "staging_name": staging_name,
        "activation_outcome_path": record.document["outcomes"]["activation_outcome_path"],
        "recovery_receipt_path": record.document["outcomes"]["activation_recovery_path"],
    }


def _load_intent(record: PlanRecord, *, required_uid: int) -> tuple[Snapshot, dict[str, Any]]:
    path = Path(record.document["outcomes"]["activation_intent_path"])
    snapshot = _snapshot(
        path,
        "AQ4 activation intent",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    document = _strict_object(snapshot.raw, "AQ4 activation intent")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 activation intent is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "plan_id",
            "plan_path",
            "plan_sha256",
            "operation_epoch",
            "created_at",
            "active_manifest_path",
            "rollback_manifest_sha256",
            "candidate_manifest_sha256",
            "staging_name",
            "activation_outcome_path",
            "recovery_receipt_path",
        },
        "AQ4 activation intent",
    )
    expected = _intent_document(record, created_at=datetime.fromtimestamp(0, timezone.utc))
    for key in expected:
        if key == "created_at":
            continue
        if document[key] != expected[key]:
            fail("AQ4 activation intent plan binding differs")
    if not isinstance(document["created_at"], str) or not document["created_at"].endswith("Z"):
        fail("AQ4 activation intent timestamp differs")
    if "/" in document["staging_name"] or not document["staging_name"]:
        fail("AQ4 activation intent staging name differs")
    return snapshot, document


def _stage_file(parent_fd: int, name: str, raw: bytes, *, required_uid: int) -> None:
    if not name or "/" in name:
        fail("AQ4 exchange staging name is invalid")
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != required_uid
            or metadata.st_nlink != 1
            or metadata.st_size != len(raw)
            or stat.S_IMODE(metadata.st_mode) != 0o644
        ):
            fail("AQ4 exchange staging metadata differs")
    except OSError as error:
        raise ActivationError("failed to stage AQ4 exchange bytes") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rename_exchange(parent_fd: int, left: str, right: str) -> None:
    try:
        _renameat2(parent_fd, left, right, RENAME_EXCHANGE)
    except OSError as error:
        raise ActivationError("AQ4 active-manifest rename exchange failed") from error


def _exchange_exact(
    parent_fd: int,
    *,
    active_name: str,
    staging_name: str,
    expected_raw: bytes,
    replacement_raw: bytes,
) -> None:
    """Exchange only exact bytes and detect a fault that occurred after rename."""

    if _entry_raw(parent_fd, active_name, "actual active manifest") != expected_raw:
        fail("active manifest bytes differ before atomic exchange")
    if _entry_raw(parent_fd, staging_name, "AQ4 exchange staging") != replacement_raw:
        fail("AQ4 exchange staging bytes differ before atomic exchange")
    exchanged = False
    try:
        _rename_exchange(parent_fd, active_name, staging_name)
        exchanged = True
    except BaseException as error:
        try:
            active = _entry_raw(parent_fd, active_name, "actual active manifest")
            staging = _entry_raw(parent_fd, staging_name, "AQ4 exchange staging")
            exchanged = active == replacement_raw and staging == expected_raw
        except Exception:
            exchanged = False
        if not exchanged:
            raise
        raise AtomicExchangeCommittedError(
            "AQ4 active-manifest exchange committed before a later fault"
        ) from error
    try:
        os.fsync(parent_fd)
        if (
            _entry_raw(parent_fd, active_name, "actual active manifest") != replacement_raw
            or _entry_raw(parent_fd, staging_name, "AQ4 exchange staging") != expected_raw
        ):
            raise ActivationError("AQ4 active-manifest exchange verification differs")
    except BaseException as error:
        if exchanged:
            raise AtomicExchangeCommittedError(
                "AQ4 active-manifest exchange committed but post-check failed"
            ) from error
        raise


def _remove_staging(parent_fd: int, name: str, expected_raw: bytes) -> None:
    try:
        if _entry_raw(parent_fd, name, "AQ4 exchange staging") != expected_raw:
            fail("AQ4 exchange staging bytes differ before removal")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as error:
        raise ActivationError("failed to remove AQ4 exchange staging") from error


def _staging_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ActivationError("failed to inspect AQ4 exchange staging") from error


def _switch_to_candidate(
    parent_fd: int,
    active: Path,
    intent: dict[str, Any],
    *,
    rollback_raw: bytes,
    candidate_raw: bytes,
    required_uid: int,
) -> None:
    staging = intent["staging_name"]
    if _staging_exists(parent_fd, staging):
        fail("AQ4 exchange staging name is unexpectedly occupied")
    _stage_file(parent_fd, staging, candidate_raw, required_uid=required_uid)
    try:
        _exchange_exact(
            parent_fd,
            active_name=active.name,
            staging_name=staging,
            expected_raw=rollback_raw,
            replacement_raw=candidate_raw,
        )
    except AtomicExchangeCommittedError:
        raise
    except BaseException:
        try:
            os.unlink(staging, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError:
            pass
        raise


def _restore_to_rollback(
    parent_fd: int,
    active: Path,
    intent: dict[str, Any],
    *,
    rollback_raw: bytes,
    candidate_raw: bytes,
    required_uid: int,
) -> bool:
    """Restore exact saved AQ4 bytes; never replace an unrecognized active file."""

    current = _entry_raw(parent_fd, active.name, "actual active manifest")
    if current == rollback_raw:
        return False
    if current != candidate_raw:
        fail("active manifest is neither exact candidate nor exact rollback bytes")
    staging = intent["staging_name"]
    if _staging_exists(parent_fd, staging):
        if _entry_raw(parent_fd, staging, "AQ4 exchange staging") != rollback_raw:
            fail("existing AQ4 exchange staging is not exact rollback bytes")
    else:
        _stage_file(parent_fd, staging, rollback_raw, required_uid=required_uid)
    try:
        _exchange_exact(
            parent_fd,
            active_name=active.name,
            staging_name=staging,
            expected_raw=candidate_raw,
            replacement_raw=rollback_raw,
        )
    except AtomicExchangeCommittedError:
        # A fault after the inverse rename is still a committed restoration.
        # Verify the exact bytes below instead of classifying it as an
        # un-restored failure solely because the post-rename caller faulted.
        pass
    if _entry_raw(parent_fd, active.name, "actual active manifest") != rollback_raw:
        fail("active manifest differs after rollback exchange")
    _remove_staging(parent_fd, staging, candidate_raw)
    return True


def _require_confirmation(
    expected_sha256: str,
    confirmation: str,
    literal: str,
) -> None:
    _hash(expected_sha256, "confirmed plan SHA-256")
    if confirmation != literal:
        fail("explicit AQ4 hardening confirmation differs")


def _forbid_unsealed_launcher(argv: list[str]) -> None:
    name = Path(argv[0]).name.lower()
    if name in {"sh", "bash", "dash", "env", "python", "python3", "python3.12"}:
        fail("reviewed operation may not invoke a shell or interpreter launcher")


def _run_operation(
    record: PlanRecord,
    stage: str,
    *,
    active: Snapshot,
    runner: CommandRunner,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    if stage not in OPERATION_STAGES:
        fail("AQ4 activation stage differs")
    operation = record.document["operations"]["stages"][stage]
    _exact(operation, {"argv", "timeout_seconds", "executable"}, f"operation {stage}")
    argv = operation["argv"]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        fail(f"operation {stage} argv differs")
    _forbid_unsealed_launcher(argv)
    _verify_seal(
        operation["executable"],
        f"operation {stage} executable",
        required_uid=operation["executable"]["uid"],
        immutable=False,
        executable=True,
    )
    environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "ULLM_AQ4_RUNTIME_HARDENING_STAGE": stage,
        "ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256": record.snapshot.sha256,
        "ULLM_AQ4_RUNTIME_HARDENING_EPOCH": record.document["operation_epoch"],
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST": os.fspath(active.path),
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256": active.sha256,
    }
    try:
        completed = runner(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=environment,
            timeout=operation["timeout_seconds"],
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ActivationError(f"AQ4 activation operation {stage} could not run") from error
    if completed.returncode != 0 and not allow_failure:
        fail(f"AQ4 activation operation {stage} failed")
    return completed


_AUDIT_ENDPOINT_CAUSES = frozenset(
    {
        "credential_unavailable",
        "deadline_elapsed",
        "endpoints_incoherent",
        "http_status",
        "invalid_response",
        "transport",
        "model_id_mismatch",
        "unavailable",
    }
)
_AUDIT_FAILURE_CAUSES = _AUDIT_ENDPOINT_CAUSES | frozenset(
    {
        "manifest_mismatch",
        "pid_not_stable",
        "process_manifest_mismatch",
        "process_unstable",
        "service_not_ready",
        "invalid_operation_output",
        "operation_unavailable",
    }
)
_AUDIT_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_AUDIT_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|jwt|session|secret|password)\b"
    r"(\s*[:=]\s*)([^\s,;\"']+)"
)
_AUDIT_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
MAX_AUDIT_STDERR_BYTES = 16 * 1024


def _sanitize_audit_stderr(value: object) -> str:
    """Retain bounded diagnostics while redacting common credential encodings."""

    raw = value if isinstance(value, str) else ""
    raw = raw.replace("\x00", "")
    raw = _AUDIT_BEARER_RE.sub(r"\1[REDACTED]", raw)
    raw = _AUDIT_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", raw)
    raw = _AUDIT_JWT_RE.sub("[REDACTED_JWT]", raw)
    encoded = raw.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_AUDIT_STDERR_BYTES:
        return raw
    return encoded[:MAX_AUDIT_STDERR_BYTES].decode("utf-8", errors="ignore") + "[truncated]"


def _audit_cause(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value in _AUDIT_FAILURE_CAUSES:
        return value
    return fallback


def _audit_endpoint_states(value: object) -> dict[str, dict[str, Any]]:
    fallback = {"ok": False, "status": None, "cause": "unavailable"}
    if not isinstance(value, dict):
        return {name: dict(fallback) for name in LIVE_ENDPOINTS}
    states: dict[str, dict[str, Any]] = {}
    for name in LIVE_ENDPOINTS:
        item = value.get(name)
        if not isinstance(item, dict):
            states[name] = dict(fallback)
            continue
        status = item.get("status")
        valid_status = status if type(status) is int and 100 <= status <= 599 else None
        ok = item.get("ok") is True
        cause = item.get("cause")
        states[name] = {
            "ok": ok,
            "status": valid_status,
            "cause": None if ok and cause is None else _audit_cause(cause, fallback="unavailable"),
        }
    return states


def _operation_failure_diagnostic(
    completed: subprocess.CompletedProcess[str] | None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    if completed is None:
        return "operation_unavailable", _audit_endpoint_states(None)
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    if len(stdout.encode("utf-8", errors="replace")) > MAX_DOCUMENT_BYTES:
        return "invalid_operation_output", _audit_endpoint_states(None)
    try:
        document = _strict_object(stdout.encode("utf-8"), "AQ4 readiness failure output")
    except ActivationError:
        return "invalid_operation_output", _audit_endpoint_states(None)
    if document.get("schema_version") != "ullm.aq4_runtime_hardening_readiness_failure.v1":
        return "invalid_operation_output", _audit_endpoint_states(None)
    return _audit_cause(document.get("cause"), fallback="invalid_operation_output"), _audit_endpoint_states(
        document.get("endpoints")
    )


def _live_proof_failure_document(
    record: PlanRecord,
    *,
    stage: str,
    active: Snapshot,
    completed: subprocess.CompletedProcess[str] | None,
    started_at: datetime,
    failed_at: datetime,
    cause: str | None = None,
) -> dict[str, Any]:
    inferred_cause, endpoints = _operation_failure_diagnostic(completed)
    stderr = "" if completed is None else _sanitize_audit_stderr(completed.stderr)
    raw_stderr = "" if completed is None or not isinstance(completed.stderr, str) else completed.stderr
    raw_stdout = "" if completed is None or not isinstance(completed.stdout, str) else completed.stdout
    returncode = None if completed is None else completed.returncode
    if type(returncode) is not int:
        returncode = None
    operation = record.document["operations"]["stages"][stage]
    return {
        "schema_version": LIVE_PROOF_AUDIT_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "stage": stage,
        "stage_status": "failed",
        "started_at": utc_timestamp(started_at),
        "failed_at": utc_timestamp(failed_at),
        "active_manifest_sha256": active.sha256,
        "operation": {
            "argv": operation["argv"],
            "executable_sha256": operation["executable"]["sha256"],
            "return_code": returncode,
            "stderr": stderr,
            "stderr_sha256": _sha256(raw_stderr.encode("utf-8", errors="replace")),
            "stdout_sha256": _sha256(raw_stdout.encode("utf-8", errors="replace")),
            "cause": _audit_cause(cause, fallback=inferred_cause),
        },
        "endpoints": endpoints,
    }


def _record_live_proof_failure(
    record: PlanRecord,
    failure: LiveProofFailure,
    *,
    required_uid: int,
    clock: Clock,
) -> Snapshot | None:
    try:
        directory = _validate_audit_directory(
            Path(record.document["outcomes"]["live_proof_audit_directory"]),
            "live_proof_audit_directory",
            required_uid=required_uid,
        )
        name = (
            f"{failure.stage}-attempt-"
            f"{utc_timestamp(clock()).replace(':', '').replace('+', '').replace('.', '')}-"
            f"{secrets.token_hex(8)}.json"
        )
        return _publish_immutable(directory / name, failure.document, required_uid=required_uid)
    except Exception:
        return None


def _live_observation(
    completed: subprocess.CompletedProcess[str],
    *,
    record: PlanRecord,
    active: Snapshot,
) -> dict[str, Any]:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    if len(stdout.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        fail("live observation output exceeds its byte bound")
    observation = _strict_object(stdout.encode("utf-8"), "AQ4 live observation")
    _exact(
        observation,
        {
            "schema_version",
            "plan_sha256",
            "operation_epoch",
            "active_manifest_sha256",
            "model_id",
            "worker_binary_path",
            "worker_binary_sha256",
            "systemd",
            "process",
            "manifest",
            "endpoints",
            "readiness",
        },
        "AQ4 live observation",
    )
    if (
        observation["schema_version"] != "ullm.aq4_runtime_hardening_live_observation.v3"
        or observation["plan_sha256"] != record.snapshot.sha256
        or observation["operation_epoch"] != record.document["operation_epoch"]
        or observation["active_manifest_sha256"] != active.sha256
        or observation["model_id"] != record.document["expected"]["model_id"]
        or observation["worker_binary_sha256"]
        != record.document["expected"]["worker_binary_sha256"]
    ):
        fail("AQ4 live observation binding differs")
    manifest_identity = _manifest_identity(active.raw, "observed active manifest")
    if observation["worker_binary_path"] != manifest_identity["worker_path"]:
        fail("AQ4 live observation worker path differs")
    systemd = _exact(
        observation["systemd"], {"unit", "active_state", "sub_state"}, "AQ4 live observation systemd"
    )
    if (
        systemd["unit"] != record.document["runtime_preconditions"]["service_unit"]
        or systemd["active_state"] != "active"
        or systemd["sub_state"] != "running"
    ):
        fail("AQ4 live observation systemd state differs")
    process = _exact(
        observation["process"],
        {"boot_id", "pid", "ppid", "starttime", "executable_sha256"},
        "AQ4 live observation process",
    )
    if (
        not isinstance(process["boot_id"], str)
        or not process["boot_id"]
        or any(type(process[name]) is not int or process[name] < 1 for name in ("pid", "ppid", "starttime"))
    ):
        fail("AQ4 live observation process identity differs")
    _hash(process["executable_sha256"], "AQ4 live observation executable SHA-256")
    manifest = _exact(
        observation["manifest"],
        {
            "active_path",
            "active_manifest_sha256",
            "file_match",
            "worker_environment_match",
            "worker_command_match",
        },
        "AQ4 live observation manifest",
    )
    if (
        manifest["active_path"] != os.fspath(active.path)
        or manifest["active_manifest_sha256"] != active.sha256
        or any(
            manifest[name] is not True
            for name in ("file_match", "worker_environment_match", "worker_command_match")
        )
    ):
        fail("AQ4 live observation manifest binding differs")
    endpoints = _exact(observation["endpoints"], set(LIVE_ENDPOINTS), "AQ4 live observation endpoints")
    for name, value in endpoints.items():
        endpoint = _exact(value, {"ok", "status", "cause"}, f"AQ4 live endpoint {name}")
        if endpoint["ok"] is not True or endpoint["status"] != 200 or endpoint["cause"] is not None:
            fail("AQ4 live observation endpoint check failed")
    readiness = _exact(
        observation["readiness"],
        {
            "timeout_seconds",
            "max_attempts",
            "attempts",
            "stable_pid_observations",
            "elapsed_milliseconds",
        },
        "AQ4 live observation readiness",
    )
    if (
        readiness["timeout_seconds"] != 120
        or readiness["max_attempts"] != 15
        or type(readiness["attempts"]) is not int
        or not 2 <= readiness["attempts"] <= readiness["max_attempts"]
        or readiness["stable_pid_observations"] != 2
        or type(readiness["elapsed_milliseconds"]) is not int
        or readiness["elapsed_milliseconds"] < 0
    ):
        fail("AQ4 live observation readiness contract differs")
    return observation


def _live_proof_document(
    record: PlanRecord,
    *,
    stage: str,
    active: Snapshot,
    observation: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    checked_at: datetime,
) -> dict[str, Any]:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    operation = record.document["operations"]["stages"][stage]
    return {
        "schema_version": LIVE_PROOF_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "stage": stage,
        "checked_at": utc_timestamp(checked_at),
        "active_manifest": {
            "path": os.fspath(active.path),
            "sha256": active.sha256,
            "bytes": active.identity.size,
        },
        "operation": {
            "argv": operation["argv"],
            "executable_sha256": operation["executable"]["sha256"],
            "stdout_sha256": _sha256(stdout.encode("utf-8")),
            "stderr_sha256": _sha256(stderr.encode("utf-8")),
        },
        "observation": observation,
    }


def _validate_live_proof(
    snapshot: Snapshot,
    record: PlanRecord,
    *,
    stage: str,
    active: Snapshot,
) -> None:
    document = _strict_object(snapshot.raw, "AQ4 live proof")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 live proof is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "plan_path",
            "plan_sha256",
            "operation_epoch",
            "stage",
            "checked_at",
            "active_manifest",
            "operation",
            "observation",
        },
        "AQ4 live proof",
    )
    if (
        document["schema_version"] != LIVE_PROOF_SCHEMA
        or document["plan_path"] != os.fspath(record.snapshot.path)
        or document["plan_sha256"] != record.snapshot.sha256
        or document["operation_epoch"] != record.document["operation_epoch"]
        or document["stage"] != stage
        or document["active_manifest"]
        != {"path": os.fspath(active.path), "sha256": active.sha256, "bytes": active.identity.size}
    ):
        fail("AQ4 live proof binding differs")
    observation = document["observation"]
    if not isinstance(observation, dict):
        fail("AQ4 live proof observation differs")
    # Reuse the same strict shape checks without treating the proof itself as a command result.
    probe = subprocess.CompletedProcess([], 0, _canonical_json(observation).decode("ascii"), "")
    _live_observation(probe, record=record, active=active)


def _run_live_stage(
    record: PlanRecord,
    stage: str,
    *,
    active: Snapshot,
    output: Path,
    runner: CommandRunner,
    required_uid: int,
    clock: Clock,
    reuse_existing: bool = False,
) -> Snapshot:
    if output.exists() or output.is_symlink():
        if not reuse_existing:
            fail("AQ4 live-proof output is already consumed")
        existing = _snapshot(
            output,
            "AQ4 live proof",
            maximum=MAX_DOCUMENT_BYTES,
            required_uid=required_uid,
            immutable=True,
        )
        _validate_live_proof(existing, record, stage=stage, active=active)
        return existing
    started_at = clock()
    try:
        completed = _run_operation(
            record,
            stage,
            active=active,
            runner=runner,
            allow_failure=True,
        )
    except ActivationError:
        raise LiveProofFailure(
            stage,
            _live_proof_failure_document(
                record,
                stage=stage,
                active=active,
                completed=None,
                started_at=started_at,
                failed_at=clock(),
                cause="operation_unavailable",
            ),
        ) from None
    if completed.returncode != 0:
        raise LiveProofFailure(
            stage,
            _live_proof_failure_document(
                record,
                stage=stage,
                active=active,
                completed=completed,
                started_at=started_at,
                failed_at=clock(),
            ),
        )
    try:
        observation = _live_observation(completed, record=record, active=active)
    except ActivationError:
        raise LiveProofFailure(
            stage,
            _live_proof_failure_document(
                record,
                stage=stage,
                active=active,
                completed=completed,
                started_at=started_at,
                failed_at=clock(),
                cause="invalid_operation_output",
            ),
        ) from None
    document = _live_proof_document(
        record,
        stage=stage,
        active=active,
        observation=observation,
        completed=completed,
        checked_at=clock(),
    )
    return _publish_immutable(output, document, required_uid=required_uid)


def _isolated_worker_observation(
    completed: subprocess.CompletedProcess[str],
    *,
    record: PlanRecord,
    candidate: Snapshot,
) -> dict[str, Any]:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    if len(stdout.encode("utf-8", errors="replace")) > MAX_DOCUMENT_BYTES:
        fail("isolated worker observation exceeds its byte bound")
    observation = _strict_object(stdout.encode("utf-8"), "AQ4 isolated worker observation")
    _exact(
        observation,
        {
            "schema_version",
            "plan_sha256",
            "operation_epoch",
            "candidate_manifest_sha256",
            "stage",
            "checked_at",
            "status",
            "cause",
            "worker",
            "operation",
            "timing",
            "cleanup",
            "production_activation_performed",
        },
        "AQ4 isolated worker observation",
    )
    if (
        observation["schema_version"]
        != "ullm.aq4_runtime_hardening_isolated_worker_observation.v1"
        or observation["plan_sha256"] != record.snapshot.sha256
        or observation["operation_epoch"] != record.document["operation_epoch"]
        or observation["candidate_manifest_sha256"] != candidate.sha256
        or observation["stage"] != ISOLATED_PREFLIGHT_STAGE
        or observation["status"] != "passed"
        or observation["cause"] is not None
        or observation["production_activation_performed"] is not False
    ):
        fail("AQ4 isolated worker observation binding differs")
    candidate_identity = _manifest_identity(candidate.raw, "candidate isolated manifest")
    worker = _exact(
        observation["worker"],
        {"model_id", "package_manifest_sha256", "device", "execution_profile"},
        "AQ4 isolated worker observation worker",
    )
    if (
        worker["model_id"] != record.document["expected"]["model_id"]
        or worker["package_manifest_sha256"] != candidate_identity["package_manifest_sha256"]
        or not isinstance(worker["device"], str)
        or not worker["device"]
        or not isinstance(worker["execution_profile"], str)
        or not worker["execution_profile"]
    ):
        fail("AQ4 isolated worker ready identity differs")
    operation = _exact(
        observation["operation"],
        {
            "argv_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "stdout_bytes",
            "stderr_bytes",
            "returncode",
        },
        "AQ4 isolated worker observation operation",
    )
    for name in ("argv_sha256", "stdout_sha256", "stderr_sha256"):
        _hash(operation[name], f"AQ4 isolated worker observation operation.{name}")
    if (
        type(operation["stdout_bytes"]) is not int
        or operation["stdout_bytes"] < 1
        or type(operation["stderr_bytes"]) is not int
        or operation["stderr_bytes"] < 0
        or type(operation["returncode"]) is not int
    ):
        fail("AQ4 isolated worker observation operation differs")
    timing = _exact(
        observation["timing"],
        {"timeout_seconds", "ready_after_milliseconds", "elapsed_milliseconds"},
        "AQ4 isolated worker observation timing",
    )
    if (
        timing["timeout_seconds"] != 120
        or type(timing["ready_after_milliseconds"]) is not int
        or timing["ready_after_milliseconds"] < 0
        or type(timing["elapsed_milliseconds"]) is not int
        or timing["elapsed_milliseconds"] < timing["ready_after_milliseconds"]
    ):
        fail("AQ4 isolated worker observation timing differs")
    cleanup = _exact(
        observation["cleanup"],
        {"terminated", "returncode"},
        "AQ4 isolated worker observation cleanup",
    )
    if cleanup["terminated"] is not True or type(cleanup["returncode"]) is not int:
        fail("AQ4 isolated worker cleanup differs")
    return observation


def _isolated_preflight_document(
    record: PlanRecord,
    *,
    candidate: Snapshot,
    observation: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    checked_at: datetime,
) -> dict[str, Any]:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    operation = record.document["operations"]["stages"][ISOLATED_PREFLIGHT_STAGE]
    return {
        "schema_version": ISOLATED_PREFLIGHT_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "stage": ISOLATED_PREFLIGHT_STAGE,
        "checked_at": utc_timestamp(checked_at),
        "candidate_manifest": {
            "path": os.fspath(candidate.path),
            "sha256": candidate.sha256,
            "bytes": candidate.identity.size,
        },
        "operation": {
            "argv": operation["argv"],
            "executable_sha256": operation["executable"]["sha256"],
            "stdout_sha256": _sha256(stdout.encode("utf-8")),
            "stderr_sha256": _sha256(stderr.encode("utf-8")),
        },
        "observation": observation,
        "production_activation_performed": False,
    }


def _validate_isolated_preflight(
    snapshot: Snapshot,
    record: PlanRecord,
    *,
    candidate: Snapshot,
) -> None:
    document = _strict_object(snapshot.raw, "AQ4 isolated candidate preflight")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 isolated candidate preflight is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "plan_path",
            "plan_sha256",
            "operation_epoch",
            "stage",
            "checked_at",
            "candidate_manifest",
            "operation",
            "observation",
            "production_activation_performed",
        },
        "AQ4 isolated candidate preflight",
    )
    if (
        document["schema_version"] != ISOLATED_PREFLIGHT_SCHEMA
        or document["plan_path"] != os.fspath(record.snapshot.path)
        or document["plan_sha256"] != record.snapshot.sha256
        or document["operation_epoch"] != record.document["operation_epoch"]
        or document["stage"] != ISOLATED_PREFLIGHT_STAGE
        or document["candidate_manifest"]
        != {"path": os.fspath(candidate.path), "sha256": candidate.sha256, "bytes": candidate.identity.size}
        or document["production_activation_performed"] is not False
    ):
        fail("AQ4 isolated candidate preflight binding differs")
    operation = _exact(
        document["operation"],
        {"argv", "executable_sha256", "stdout_sha256", "stderr_sha256"},
        "AQ4 isolated candidate preflight operation",
    )
    expected = record.document["operations"]["stages"][ISOLATED_PREFLIGHT_STAGE]
    if operation["argv"] != expected["argv"] or operation["executable_sha256"] != expected["executable"]["sha256"]:
        fail("AQ4 isolated candidate preflight operation binding differs")
    _hash(operation["stdout_sha256"], "AQ4 isolated candidate preflight stdout SHA-256")
    _hash(operation["stderr_sha256"], "AQ4 isolated candidate preflight stderr SHA-256")
    observation = document["observation"]
    if not isinstance(observation, dict):
        fail("AQ4 isolated candidate preflight observation differs")
    probe = subprocess.CompletedProcess([], 0, _canonical_json(observation).decode("ascii"), "")
    _isolated_worker_observation(probe, record=record, candidate=candidate)


def _load_isolated_preflight(
    record: PlanRecord,
    *,
    candidate: Snapshot,
    required_uid: int,
) -> Snapshot:
    path = Path(record.document["outcomes"]["candidate_isolated_preflight_path"])
    if not path.exists() and not path.is_symlink():
        fail("candidate isolated worker preflight is not recorded")
    snapshot = _snapshot(
        path,
        "AQ4 isolated candidate preflight",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    _validate_isolated_preflight(snapshot, record, candidate=candidate)
    return snapshot


def run_isolated_candidate_preflight(
    record: PlanRecord,
    *,
    required_uid: int = 0,
    runner: CommandRunner = subprocess.run,
    clock: Clock = utc_now,
) -> Snapshot:
    """Run only the candidate worker and publish a reusable immutable receipt."""

    _active, _rollback, candidate = _verify_plan_inputs(record, required_uid=required_uid)
    path = Path(record.document["outcomes"]["candidate_isolated_preflight_path"])
    if path.exists() or path.is_symlink():
        return _load_isolated_preflight(record, candidate=candidate, required_uid=required_uid)
    completed = _run_operation(
        record,
        ISOLATED_PREFLIGHT_STAGE,
        active=candidate,
        runner=runner,
        allow_failure=True,
    )
    if completed.returncode != 0:
        fail("candidate isolated worker preflight failed")
    observation = _isolated_worker_observation(completed, record=record, candidate=candidate)
    document = _isolated_preflight_document(
        record,
        candidate=candidate,
        observation=observation,
        completed=completed,
        checked_at=clock(),
    )
    return _publish_immutable(path, document, required_uid=required_uid)


def _proof_reference(snapshot: Snapshot) -> dict[str, Any]:
    return {"path": os.fspath(snapshot.path), "sha256": snapshot.sha256}


def _outcome_document(
    record: PlanRecord,
    *,
    intent: Snapshot,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    failure_stage: str | None,
    stages: dict[str, str],
    observed_active_sha256: str | None,
    candidate_proof: Snapshot | None,
    rollback_proof: Snapshot | None,
    restoration_attempted: bool,
    legacy_assets_checked: bool,
) -> dict[str, Any]:
    if status not in {"activated", "failed_restored", "failed_restore", "aborted_before_swap"}:
        fail("AQ4 activation outcome status differs")
    return {
        "schema_version": OUTCOME_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "plan_id": record.document["plan_id"],
        "intent_path": os.fspath(intent.path),
        "intent_sha256": intent.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "started_at": utc_timestamp(started_at),
        "completed_at": utc_timestamp(completed_at),
        "status": status,
        "failure_stage": failure_stage,
        "stages": stages,
        "rollback_manifest_sha256": record.document["rollback_manifest"]["sha256"],
        "candidate_manifest_sha256": record.document["candidate_runtime"]["manifest"]["sha256"],
        "observed_active_manifest_sha256": observed_active_sha256,
        "candidate_live_proof": None if candidate_proof is None else _proof_reference(candidate_proof),
        "rollback_live_proof": None if rollback_proof is None else _proof_reference(rollback_proof),
        "restoration": {
            "attempted": restoration_attempted,
            "legacy_assets_checked": legacy_assets_checked,
        },
    }


def _load_activation_outcome(
    record: PlanRecord,
    *,
    required_uid: int,
) -> tuple[Snapshot, dict[str, Any]]:
    path = Path(record.document["outcomes"]["activation_outcome_path"])
    snapshot = _snapshot(
        path,
        "AQ4 activation outcome",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    document = _strict_object(snapshot.raw, "AQ4 activation outcome")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 activation outcome is not canonical JSON")
    required = {
        "schema_version",
        "plan_path",
        "plan_sha256",
        "plan_id",
        "intent_path",
        "intent_sha256",
        "operation_epoch",
        "started_at",
        "completed_at",
        "status",
        "failure_stage",
        "stages",
        "rollback_manifest_sha256",
        "candidate_manifest_sha256",
        "observed_active_manifest_sha256",
        "candidate_live_proof",
        "rollback_live_proof",
        "restoration",
    }
    _exact(document, required, "AQ4 activation outcome")
    if (
        document["schema_version"] != OUTCOME_SCHEMA
        or document["plan_path"] != os.fspath(record.snapshot.path)
        or document["plan_sha256"] != record.snapshot.sha256
        or document["plan_id"] != record.document["plan_id"]
        or document["operation_epoch"] != record.document["operation_epoch"]
        or document["rollback_manifest_sha256"] != record.document["rollback_manifest"]["sha256"]
        or document["candidate_manifest_sha256"]
        != record.document["candidate_runtime"]["manifest"]["sha256"]
        or document["status"]
        not in {"activated", "failed_restored", "failed_restore", "aborted_before_swap"}
    ):
        fail("AQ4 activation outcome plan binding differs")
    intent, _ = _load_intent(record, required_uid=required_uid)
    if document["intent_path"] != os.fspath(intent.path) or document["intent_sha256"] != intent.sha256:
        fail("AQ4 activation outcome intent binding differs")
    return snapshot, document


def _active_snapshot(record: PlanRecord, *, required_uid: int) -> Snapshot:
    binding = record.document["active_manifest"]
    return _snapshot(
        Path(binding["path"]),
        "active manifest",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=False,
    )


def _recovery_inputs(record: PlanRecord, *, required_uid: int) -> Snapshot:
    """Re-pin inputs needed to safely restore AQ4 without relying on candidate files."""

    _verify_git_source(
        record.document["control_source"], "activation control source", required_uid=required_uid
    )
    _verify_operations(record.document["operations"], required_uid=required_uid)
    _verify_preconditions(record.document["runtime_preconditions"], required_uid=required_uid)
    lock = _lock_metadata(Path(record.document["lock"]["path"]), required_uid=required_uid)
    if lock != record.document["lock"]:
        fail("AQ4 activation lock seal drifted")
    rollback = _verify_seal(
        record.document["rollback_manifest"],
        "saved rollback manifest",
        required_uid=required_uid,
        immutable=True,
    )
    _verify_legacy_runtime(record.document["legacy_runtime"], rollback)
    return rollback


def _record_attempt(
    record: PlanRecord,
    *,
    kind: str,
    error: BaseException,
    active_sha256: str | None,
    directory_key: str,
    required_uid: int,
    clock: Clock,
) -> Snapshot | None:
    directory = Path(record.document["outcomes"][directory_key])
    try:
        _validate_audit_directory(directory, directory_key, required_uid=required_uid)
        name = f"{kind}-attempt-{utc_timestamp(clock()).replace(':', '').replace('+', '').replace('.', '')}-{secrets.token_hex(8)}.json"
        document = {
            "schema_version": ATTEMPT_SCHEMA,
            "plan_path": os.fspath(record.snapshot.path),
            "plan_sha256": record.snapshot.sha256,
            "operation_epoch": record.document["operation_epoch"],
            "kind": kind,
            "recorded_at": utc_timestamp(clock()),
            "active_manifest_sha256": active_sha256,
            "error": type(error).__name__,
        }
        return _publish_immutable(directory / name, document, required_uid=required_uid)
    except Exception:
        return None


def _published_outcome_or_none(
    path: Path,
    document: dict[str, Any],
    *,
    required_uid: int,
) -> Snapshot | None:
    raw = _canonical_json(document)
    try:
        return _publish_immutable(path, document, required_uid=required_uid)
    except ImmutablePublicationCommittedError:
        return _committed_immutable(path, raw, required_uid=required_uid)


def _restore_and_prove(
    record: PlanRecord,
    intent_document: dict[str, Any],
    *,
    parent_fd: int,
    active_path: Path,
    rollback: Snapshot,
    candidate: Snapshot,
    runner: CommandRunner,
    required_uid: int,
    clock: Clock,
    reuse_existing_proof: bool,
) -> tuple[Snapshot, Snapshot | None, bool]:
    """Restore exact bytes, reconcile, then obtain a legacy live proof."""

    _restore_to_rollback(
        parent_fd,
        active_path,
        intent_document,
        rollback_raw=rollback.raw,
        candidate_raw=candidate.raw,
        required_uid=required_uid,
    )
    active = _active_snapshot(record, required_uid=required_uid)
    if active.raw != rollback.raw:
        fail("exact rollback bytes are not active after restoration")
    _run_operation(record, "rollback_reconcile", active=active, runner=runner)
    _verify_legacy_runtime(record.document["legacy_runtime"], rollback)
    proof = _run_live_stage(
        record,
        "rollback_live_proof",
        active=active,
        output=Path(record.document["outcomes"]["rollback_live_proof_path"]),
        runner=runner,
        required_uid=required_uid,
        clock=clock,
        reuse_existing=reuse_existing_proof,
    )
    return active, proof, True


def execute_activation(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    confirmation: str,
    required_uid: int = 0,
    runner: CommandRunner = subprocess.run,
    clock: Clock = utc_now,
    fault_hook: FaultHook | None = None,
) -> ExecutionResult:
    """Execute one human-confirmed AQ4-to-AQ4 transition under the shared lock."""

    _require_confirmation(expected_plan_sha256, confirmation, ACTIVATION_CONFIRMATION)
    initial = load_plan(plan_path, required_uid=required_uid)
    if initial.snapshot.sha256 != expected_plan_sha256:
        fail("confirmed AQ4 hardening plan SHA-256 differs")
    outcomes = initial.document["outcomes"]
    _output_unused(Path(outcomes["activation_intent_path"]), "activation intent")
    _output_unused(Path(outcomes["activation_outcome_path"]), "activation outcome")
    lock_fd = -1
    parent_fd = -1
    record: PlanRecord | None = None
    intent_snapshot: Snapshot | None = None
    intent_document: dict[str, Any] | None = None
    rollback: Snapshot | None = None
    candidate: Snapshot | None = None
    candidate_proof: Snapshot | None = None
    rollback_proof: Snapshot | None = None
    switched = False
    outcome_committed = False
    stages = {name: "pending" for name in STAGES}
    failure_stage: str | None = None
    started_at = clock()
    try:
        lock_fd = _open_locked_activation_lock(initial, required_uid=required_uid)
        record = load_plan(plan_path, required_uid=required_uid)
        _require_same_plan(initial, record, expected_plan_sha256)
        active, rollback, candidate = _verify_plan_inputs(record, required_uid=required_uid)
        _load_isolated_preflight(record, candidate=candidate, required_uid=required_uid)
        _output_unused(Path(record.document["outcomes"]["activation_intent_path"]), "activation intent")
        _output_unused(Path(record.document["outcomes"]["activation_outcome_path"]), "activation outcome")
        active_path = active.path
        parent_fd = _open_parent(active_path, "active manifest", required_uid=required_uid)
        intent_document = _intent_document(record, created_at=clock())
        try:
            intent_snapshot = _publish_immutable(
                Path(record.document["outcomes"]["activation_intent_path"]),
                intent_document,
                required_uid=required_uid,
            )
        except ImmutablePublicationCommittedError:
            intent_snapshot, _loaded = _load_intent(record, required_uid=required_uid)
        if fault_hook is not None:
            fault_hook("after_intent")
        # Re-seal credentials and source/runtime inputs after intent, immediately before swap.
        active, rollback, candidate = _verify_plan_inputs(record, required_uid=required_uid)
        _load_isolated_preflight(record, candidate=candidate, required_uid=required_uid)
        try:
            _switch_to_candidate(
                parent_fd,
                active_path,
                intent_document,
                rollback_raw=rollback.raw,
                candidate_raw=candidate.raw,
                required_uid=required_uid,
            )
            switched = True
        except AtomicExchangeCommittedError:
            switched = True
            raise
        if fault_hook is not None:
            fault_hook("after_swap")
        active = _active_snapshot(record, required_uid=required_uid)
        if active.raw != candidate.raw:
            fail("candidate bytes are not active after exchange")
        _verify_protected_runtime(record.document["candidate_runtime"], required_uid=required_uid)
        failure_stage = "candidate_reconcile"
        _run_operation(record, "candidate_reconcile", active=active, runner=runner)
        stages["candidate_reconcile"] = "passed"
        active = _active_snapshot(record, required_uid=required_uid)
        if active.raw != candidate.raw:
            fail("active manifest drifted during candidate reconciliation")
        failure_stage = "candidate_live_proof"
        candidate_proof = _run_live_stage(
            record,
            "candidate_live_proof",
            active=active,
            output=Path(record.document["outcomes"]["candidate_live_proof_path"]),
            runner=runner,
            required_uid=required_uid,
            clock=clock,
        )
        stages["candidate_live_proof"] = "passed"
        _verify_protected_runtime(record.document["candidate_runtime"], required_uid=required_uid)
        _remove_staging(parent_fd, intent_document["staging_name"], rollback.raw)
        if fault_hook is not None:
            fault_hook("before_outcome_publication")
        stages["rollback_reconcile"] = "skipped"
        stages["rollback_live_proof"] = "skipped"
        failure_stage = "outcome_publication"
        assert intent_snapshot is not None
        outcome = _outcome_document(
            record,
            intent=intent_snapshot,
            started_at=started_at,
            completed_at=clock(),
            status="activated",
            failure_stage=None,
            stages=stages,
            observed_active_sha256=active.sha256,
            candidate_proof=candidate_proof,
            rollback_proof=None,
            restoration_attempted=False,
            legacy_assets_checked=False,
        )
        outcome_path = Path(record.document["outcomes"]["activation_outcome_path"])
        published = _published_outcome_or_none(
            outcome_path, outcome, required_uid=required_uid
        )
        if published is None:
            raise ActivationError("activation outcome publication did not commit")
        # This immutable receipt is the commit boundary.  Do not add fallible
        # source checks, process checks, or output operations after this point.
        outcome_committed = True
        return ExecutionResult(published.path, published.sha256, "activated")
    except Exception as error:
        if record is None or intent_snapshot is None or intent_document is None:
            raise
        if outcome_committed:
            raise
        if rollback is None or candidate is None:
            raise
        if failure_stage in STAGES and stages[failure_stage] == "pending":
            stages[failure_stage] = "failed"
        if isinstance(error, LiveProofFailure):
            _record_live_proof_failure(
                record,
                error,
                required_uid=required_uid,
                clock=clock,
            )
        active_path = Path(record.document["active_manifest"]["path"])
        observed_hash: str | None = None
        restoration_attempted = False
        legacy_checked = False
        if switched and parent_fd >= 0:
            restoration_attempted = True
            try:
                restored, rollback_proof, legacy_checked = _restore_and_prove(
                    record,
                    intent_document,
                    parent_fd=parent_fd,
                    active_path=active_path,
                    rollback=rollback,
                    candidate=candidate,
                    runner=runner,
                    required_uid=required_uid,
                    clock=clock,
                    reuse_existing_proof=False,
                )
                observed_hash = restored.sha256
                stages["rollback_reconcile"] = "passed"
                stages["rollback_live_proof"] = "passed"
                status = "failed_restored"
            except Exception as restore_error:
                if isinstance(restore_error, LiveProofFailure):
                    _record_live_proof_failure(
                        record,
                        restore_error,
                        required_uid=required_uid,
                        clock=clock,
                    )
                    if stages[restore_error.stage] == "pending":
                        stages[restore_error.stage] = "failed"
                try:
                    observed_hash = _active_snapshot(record, required_uid=required_uid).sha256
                except Exception:
                    observed_hash = None
                if stages["rollback_reconcile"] == "pending":
                    stages["rollback_reconcile"] = "failed"
                if stages["rollback_live_proof"] == "pending":
                    stages["rollback_live_proof"] = "failed"
                status = "failed_restore"
        else:
            try:
                observed_hash = _active_snapshot(record, required_uid=required_uid).sha256
            except Exception:
                observed_hash = None
            status = "aborted_before_swap"
        for name in STAGES:
            if stages[name] == "pending":
                stages[name] = "not_run"
        outcome = _outcome_document(
            record,
            intent=intent_snapshot,
            started_at=started_at,
            completed_at=clock(),
            status=status,
            failure_stage=failure_stage or "before_swap",
            stages=stages,
            observed_active_sha256=observed_hash,
            candidate_proof=candidate_proof,
            rollback_proof=rollback_proof,
            restoration_attempted=restoration_attempted,
            legacy_assets_checked=legacy_checked,
        )
        try:
            _published_outcome_or_none(
                Path(record.document["outcomes"]["activation_outcome_path"]),
                outcome,
                required_uid=required_uid,
            )
        except Exception:
            pass
        raise ActivationError("AQ4 hardening activation failed; inspect immutable outcome/recovery") from error
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def _audit_exists(directory: Path) -> bool:
    try:
        return any(item.is_file() and not item.is_symlink() for item in directory.iterdir())
    except OSError:
        return False


def _recovery_document(
    record: PlanRecord,
    *,
    intent: Snapshot,
    reason: str,
    rollback_proof: Snapshot,
    completed_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "plan_id": record.document["plan_id"],
        "intent_path": os.fspath(intent.path),
        "intent_sha256": intent.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "reason": reason,
        "completed_at": utc_timestamp(completed_at),
        "rollback_manifest_sha256": record.document["rollback_manifest"]["sha256"],
        "rollback_live_proof": _proof_reference(rollback_proof),
        "status": "recovered",
    }


def _rollback_document(
    record: PlanRecord,
    *,
    intent: Snapshot,
    rollback_proof: Snapshot,
    completed_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": ROLLBACK_SCHEMA,
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "plan_id": record.document["plan_id"],
        "intent_path": os.fspath(intent.path),
        "intent_sha256": intent.sha256,
        "operation_epoch": record.document["operation_epoch"],
        "completed_at": utc_timestamp(completed_at),
        "candidate_manifest_sha256": record.document["candidate_runtime"]["manifest"]["sha256"],
        "rollback_manifest_sha256": record.document["rollback_manifest"]["sha256"],
        "rollback_live_proof": _proof_reference(rollback_proof),
        "status": "rolled_back",
    }


def _load_completed_recovery(
    record: PlanRecord,
    *,
    required_uid: int,
) -> Snapshot | None:
    path = Path(record.document["outcomes"]["activation_recovery_path"])
    if not path.exists() and not path.is_symlink():
        return None
    snapshot = _snapshot(
        path,
        "AQ4 activation recovery receipt",
        maximum=MAX_DOCUMENT_BYTES,
        required_uid=required_uid,
        immutable=True,
    )
    document = _strict_object(snapshot.raw, "AQ4 activation recovery receipt")
    if _canonical_json(document) != snapshot.raw:
        fail("AQ4 activation recovery receipt is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "plan_path",
            "plan_sha256",
            "plan_id",
            "intent_path",
            "intent_sha256",
            "operation_epoch",
            "reason",
            "completed_at",
            "rollback_manifest_sha256",
            "rollback_live_proof",
            "status",
        },
        "AQ4 activation recovery receipt",
    )
    if (
        document["schema_version"] != RECOVERY_SCHEMA
        or document["plan_path"] != os.fspath(record.snapshot.path)
        or document["plan_sha256"] != record.snapshot.sha256
        or document["status"] != "recovered"
    ):
        fail("AQ4 activation recovery receipt plan binding differs")
    return snapshot


def execute_activation_recovery(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    confirmation: str,
    required_uid: int = 0,
    runner: CommandRunner = subprocess.run,
    clock: Clock = utc_now,
) -> ExecutionResult:
    """Recover an intent-only crash, failed restore, or incomplete rollback.

    Failed attempts are only recorded in unique audit files.  The fixed
    immutable recovery receipt is published only after exact AQ4 bytes and the
    rollback live proof both succeed, so it remains available for a retry.
    """

    _require_confirmation(expected_plan_sha256, confirmation, RECOVERY_CONFIRMATION)
    initial = load_plan(plan_path, required_uid=required_uid)
    if initial.snapshot.sha256 != expected_plan_sha256:
        fail("confirmed AQ4 hardening plan SHA-256 differs")
    lock_fd = -1
    parent_fd = -1
    record: PlanRecord | None = None
    active_hash: str | None = None
    try:
        lock_fd = _open_locked_activation_lock(initial, required_uid=required_uid)
        record = load_plan(plan_path, required_uid=required_uid)
        _require_same_plan(initial, record, expected_plan_sha256)
        existing = _load_completed_recovery(record, required_uid=required_uid)
        if existing is not None:
            return ExecutionResult(existing.path, existing.sha256, "recovered")
        intent, intent_document = _load_intent(record, required_uid=required_uid)
        outcome_path = Path(record.document["outcomes"]["activation_outcome_path"])
        reason = "intent_incomplete"
        if outcome_path.exists() or outcome_path.is_symlink():
            _unused, outcome = _load_activation_outcome(record, required_uid=required_uid)
            if outcome["status"] == "activated":
                rollback_audits = Path(record.document["outcomes"]["rollback_audit_directory"])
                if not _audit_exists(rollback_audits):
                    fail("successful activation has no incomplete rollback to recover")
                reason = "rollback_incomplete"
            elif outcome["status"] == "failed_restore":
                reason = "failed_restore"
            elif outcome["status"] == "failed_restored":
                reason = "failed_restored_proof_recheck"
            else:
                reason = "aborted_before_swap"
        rollback = _recovery_inputs(record, required_uid=required_uid)
        active = _active_snapshot(record, required_uid=required_uid)
        active_hash = active.sha256
        candidate_hash = record.document["candidate_runtime"]["manifest"]["sha256"]
        if active.raw == rollback.raw:
            candidate = active
        elif active.sha256 == candidate_hash:
            candidate = active
        else:
            fail("recovery active bytes do not match exact candidate or rollback")
        parent_fd = _open_parent(active.path, "active manifest", required_uid=required_uid)
        restored, proof, _legacy = _restore_and_prove(
            record,
            intent_document,
            parent_fd=parent_fd,
            active_path=active.path,
            rollback=rollback,
            candidate=candidate,
            runner=runner,
            required_uid=required_uid,
            clock=clock,
            reuse_existing_proof=True,
        )
        active_hash = restored.sha256
        assert proof is not None
        receipt = _recovery_document(
            record,
            intent=intent,
            reason=reason,
            rollback_proof=proof,
            completed_at=clock(),
        )
        path = Path(record.document["outcomes"]["activation_recovery_path"])
        published = _published_outcome_or_none(path, receipt, required_uid=required_uid)
        if published is None:
            fail("AQ4 recovery receipt publication did not commit")
        return ExecutionResult(published.path, published.sha256, "recovered")
    except Exception as error:
        if record is not None:
            if isinstance(error, LiveProofFailure):
                _record_live_proof_failure(
                    record,
                    error,
                    required_uid=required_uid,
                    clock=clock,
                )
            _record_attempt(
                record,
                kind="recovery",
                error=error,
                active_sha256=active_hash,
                directory_key="recovery_audit_directory",
                required_uid=required_uid,
                clock=clock,
            )
        raise ActivationError("AQ4 hardening recovery is incomplete; retry uses a new audit") from error
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def execute_rollback(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    confirmation: str,
    required_uid: int = 0,
    runner: CommandRunner = subprocess.run,
    clock: Clock = utc_now,
) -> ExecutionResult:
    """Manually roll back only exact candidate-active bytes under the same lock."""

    _require_confirmation(expected_plan_sha256, confirmation, ROLLBACK_CONFIRMATION)
    initial = load_plan(plan_path, required_uid=required_uid)
    if initial.snapshot.sha256 != expected_plan_sha256:
        fail("confirmed AQ4 hardening plan SHA-256 differs")
    lock_fd = -1
    parent_fd = -1
    record: PlanRecord | None = None
    active_hash: str | None = None
    try:
        lock_fd = _open_locked_activation_lock(initial, required_uid=required_uid)
        record = load_plan(plan_path, required_uid=required_uid)
        _require_same_plan(initial, record, expected_plan_sha256)
        rollback_path = Path(record.document["outcomes"]["rollback_outcome_path"])
        _output_unused(rollback_path, "manual rollback outcome")
        _unused, outcome = _load_activation_outcome(record, required_uid=required_uid)
        if outcome["status"] != "activated":
            fail("manual rollback requires an immutable successful activation receipt")
        intent, intent_document = _load_intent(record, required_uid=required_uid)
        rollback = _recovery_inputs(record, required_uid=required_uid)
        candidate = _verify_seal(
            record.document["candidate_runtime"]["manifest"],
            "candidate frozen manifest",
            required_uid=required_uid,
            immutable=True,
        )
        active = _active_snapshot(record, required_uid=required_uid)
        active_hash = active.sha256
        if active.raw != candidate.raw:
            fail("manual rollback requires exact candidate-active bytes")
        parent_fd = _open_parent(active.path, "active manifest", required_uid=required_uid)
        restored, proof, _legacy = _restore_and_prove(
            record,
            intent_document,
            parent_fd=parent_fd,
            active_path=active.path,
            rollback=rollback,
            candidate=candidate,
            runner=runner,
            required_uid=required_uid,
            clock=clock,
            reuse_existing_proof=False,
        )
        active_hash = restored.sha256
        assert proof is not None
        receipt = _rollback_document(
            record,
            intent=intent,
            rollback_proof=proof,
            completed_at=clock(),
        )
        published = _published_outcome_or_none(
            rollback_path, receipt, required_uid=required_uid
        )
        if published is None:
            fail("manual rollback receipt publication did not commit")
        return ExecutionResult(published.path, published.sha256, "rolled_back")
    except Exception as error:
        if record is not None:
            if isinstance(error, LiveProofFailure):
                _record_live_proof_failure(
                    record,
                    error,
                    required_uid=required_uid,
                    clock=clock,
                )
            _record_attempt(
                record,
                kind="rollback",
                error=error,
                active_sha256=active_hash,
                directory_key="rollback_audit_directory",
                required_uid=required_uid,
                clock=clock,
            )
        raise ActivationError("AQ4 hardening rollback is incomplete; use locked recovery") from error
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
