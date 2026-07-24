#!/usr/bin/env python3
"""Seal immutable runtime files and their trusted pathname ancestry."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from served_model_active_binding import FileIdentity, StableFileSnapshot


READ_CHUNK_BYTES = 64 * 1024
POSIX_ACL_XATTRS = ("system.posix_acl_access", "system.posix_acl_default")
FORBIDDEN_SECURITY_XATTRS = ("security.capability",)
MAX_RUNTIME_TREE_ENTRIES = 250_000
MAX_RUNTIME_TREE_TOTAL_BYTES = 256 * 1024 * 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 4_096


class RuntimeArtifactSealError(RuntimeError):
    """A runtime artifact is mutable by an identity outside the executor."""


@dataclass(frozen=True, slots=True)
class RuntimeDirectoryIdentity:
    """Security-relevant identity of one component in an artifact parent tree."""

    path: Path
    device: int
    inode: int
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True, slots=True)
class RuntimeArtifactSeal:
    """Exact file bytes/metadata plus a stable, non-writable parent tree."""

    label: str
    required_uid: int
    maximum: int
    snapshot: StableFileSnapshot
    ancestry: tuple[RuntimeDirectoryIdentity, ...]


@dataclass(frozen=True, slots=True)
class RuntimeTreeEntry:
    """One recursively sealed runtime-tree entry."""

    relative_path: str
    device: int
    inode: int
    ctime_ns: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class RuntimeTreeSeal:
    """Metadata seal for a complete tokenizer or product payload tree."""

    label: str
    required_uid: int
    root: Path
    ancestry: tuple[RuntimeDirectoryIdentity, ...]
    entries: tuple[RuntimeTreeEntry, ...]
    fingerprint_sha256: str


def _lexical_absolute(path: Path) -> Path:
    if not isinstance(path, Path):
        raise RuntimeArtifactSealError("runtime artifact path is not pathlib")
    raw = os.fspath(path)
    if (
        not raw
        or "\x00" in raw
        or not path.is_absolute()
        or path.anchor != "/"
        or raw.startswith("//")
        or os.path.normpath(raw) != raw
        or path.name in {"", ".", ".."}
        or ".." in path.parts
    ):
        raise RuntimeArtifactSealError(
            "runtime artifact path is not lexical absolute"
        )
    return path


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeArtifactSealError(
            "runtime artifact directory safety flags are unavailable"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeArtifactSealError(
            "runtime artifact file safety flags are unavailable"
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _has_posix_acl(descriptor: int) -> bool:
    for attribute in POSIX_ACL_XATTRS:
        try:
            os.getxattr(descriptor, attribute)
        except OSError as error:
            if error.errno in {
                errno.ENODATA,
                getattr(errno, "ENOATTR", errno.ENODATA),
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                continue
            raise RuntimeArtifactSealError(
                "runtime artifact ACL metadata cannot be inspected"
            ) from error
        else:
            return True
    return False


def _has_forbidden_security_xattr(descriptor: int) -> bool:
    for attribute in FORBIDDEN_SECURITY_XATTRS:
        try:
            os.getxattr(descriptor, attribute)
        except OSError as error:
            if error.errno in {
                errno.ENODATA,
                getattr(errno, "ENOATTR", errno.ENODATA),
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                continue
            raise RuntimeArtifactSealError(
                "runtime artifact security metadata cannot be inspected"
            ) from error
        else:
            return True
    return False


def _directory_identity(
    path: Path,
    descriptor: int,
    *,
    required_uid: int,
    below_sticky_tmp: bool,
) -> tuple[RuntimeDirectoryIdentity, bool]:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry cannot be inspected"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry is not directories"
        )
    if _has_posix_acl(descriptor):
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry has a POSIX ACL"
        )
    if _has_forbidden_security_xattr(descriptor):
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry has a file capability"
        )
    shared_writable = bool(stat.S_IMODE(metadata.st_mode) & 0o022)
    sticky_tmp = (
        path == Path("/tmp")
        and metadata.st_uid == 0
        and bool(metadata.st_mode & stat.S_ISVTX)
        and shared_writable
    )
    if shared_writable and not sticky_tmp:
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry is group/world writable"
        )
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry has privilege mode bits"
        )
    if below_sticky_tmp:
        if metadata.st_uid != required_uid:
            raise RuntimeArtifactSealError(
                "runtime artifact ancestry below /tmp has another owner"
            )
    elif metadata.st_uid not in {0, required_uid}:
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry owner is untrusted"
        )
    return (
        RuntimeDirectoryIdentity(
            path=path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            uid=metadata.st_uid,
            gid=metadata.st_gid,
        ),
        below_sticky_tmp or sticky_tmp,
    )


def _open_directory_path(
    directory: Path,
    *,
    required_uid: int,
) -> tuple[int, tuple[RuntimeDirectoryIdentity, ...]]:
    descriptor = -1
    identities: list[RuntimeDirectoryIdentity] = []
    try:
        descriptor = os.open("/", _directory_flags())
        current = Path("/")
        below_sticky_tmp = False
        identity, below_sticky_tmp = _directory_identity(
            current,
            descriptor,
            required_uid=required_uid,
            below_sticky_tmp=below_sticky_tmp,
        )
        identities.append(identity)
        for component in directory.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            current /= component
            identity, below_sticky_tmp = _directory_identity(
                current,
                descriptor,
                required_uid=required_uid,
                below_sticky_tmp=below_sticky_tmp,
            )
            identities.append(identity)
        return descriptor, tuple(identities)
    except RuntimeArtifactSealError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise RuntimeArtifactSealError(
            "runtime artifact ancestry is unavailable or traverses a symlink"
        ) from error


def _open_parent_tree(
    path: Path,
    *,
    required_uid: int,
) -> tuple[int, tuple[RuntimeDirectoryIdentity, ...]]:
    return _open_directory_path(path.parent, required_uid=required_uid)


def _read_all(descriptor: int, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(
                descriptor,
                min(READ_CHUNK_BYTES, maximum - total + 1),
            )
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > maximum:
                raise RuntimeArtifactSealError(
                    "runtime artifact exceeds its byte bound"
                )
            chunks.append(chunk)
    except RuntimeArtifactSealError:
        raise
    except OSError as error:
        raise RuntimeArtifactSealError(
            "runtime artifact cannot be read"
        ) from error


def _scan(
    path: Path,
    *,
    label: str,
    maximum: int,
    required_uid: int,
) -> RuntimeArtifactSeal:
    parent_descriptor, ancestry = _open_parent_tree(
        path,
        required_uid=required_uid,
    )
    descriptor = -1
    verification_parent = -1
    try:
        try:
            entry_before = FileIdentity.from_stat(
                os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            descriptor = os.open(
                path.name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise RuntimeArtifactSealError(
                "runtime artifact is unavailable or is a symlink"
            ) from error
        opened = FileIdentity.from_stat(os.fstat(descriptor))
        if entry_before != opened:
            raise RuntimeArtifactSealError(
                "runtime artifact changed while it was opened"
            )
        if (
            not stat.S_ISREG(opened.mode)
            or opened.uid != required_uid
            or opened.links != 1
            or opened.size < 1
            or opened.size > maximum
            or stat.S_IMODE(opened.mode) & 0o022
            or opened.mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise RuntimeArtifactSealError(
                "runtime artifact file metadata is unsafe"
            )
        if _has_posix_acl(descriptor):
            raise RuntimeArtifactSealError(
                "runtime artifact file has a POSIX ACL"
            )
        if _has_forbidden_security_xattr(descriptor):
            raise RuntimeArtifactSealError(
                "runtime artifact file has a file capability"
            )
        raw = _read_all(descriptor, maximum=maximum)
        opened_after = FileIdentity.from_stat(os.fstat(descriptor))
        entry_after = FileIdentity.from_stat(
            os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        )
        verification_parent, ancestry_after = _open_parent_tree(
            path,
            required_uid=required_uid,
        )
        if (
            opened != opened_after
            or opened != entry_after
            or len(raw) != opened.size
            or ancestry != ancestry_after
        ):
            raise RuntimeArtifactSealError(
                "runtime artifact or its parent tree changed while sealing"
            )
        return RuntimeArtifactSeal(
            label=label,
            required_uid=required_uid,
            maximum=maximum,
            snapshot=StableFileSnapshot(
                path=path,
                raw=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
                identity=opened,
            ),
            ancestry=ancestry,
        )
    except RuntimeArtifactSealError:
        raise
    except OSError as error:
        raise RuntimeArtifactSealError(
            "runtime artifact cannot be inspected safely"
        ) from error
    finally:
        for value in (verification_parent, descriptor, parent_descriptor):
            if value >= 0:
                try:
                    os.close(value)
                except OSError:
                    pass


def capture_runtime_artifact_seal(
    path: Path,
    *,
    label: str,
    maximum: int,
    required_uid: int,
) -> RuntimeArtifactSeal:
    """Capture an artifact only when two complete security scans agree."""

    if (
        not isinstance(label, str)
        or not label
        or "\x00" in label
        or type(maximum) is not int
        or maximum < 1
        or type(required_uid) is not int
        or required_uid < 0
    ):
        raise RuntimeArtifactSealError(
            "runtime artifact seal parameters are invalid"
        )
    selected = _lexical_absolute(path)
    first = _scan(
        selected,
        label=label,
        maximum=maximum,
        required_uid=required_uid,
    )
    second = _scan(
        selected,
        label=label,
        maximum=maximum,
        required_uid=required_uid,
    )
    if first != second:
        raise RuntimeArtifactSealError(
            "runtime artifact changed while sealing"
        )
    return first


def require_runtime_artifact_seal(
    expected: RuntimeArtifactSeal,
    *,
    required_uid: int,
) -> RuntimeArtifactSeal:
    """Require exact file bytes/metadata and the same trusted parent tree."""

    if expected.required_uid != required_uid:
        raise RuntimeArtifactSealError(
            "runtime artifact seal executor differs"
        )
    observed = capture_runtime_artifact_seal(
        expected.snapshot.path,
        label=expected.label,
        maximum=expected.maximum,
        required_uid=required_uid,
    )
    if observed != expected:
        raise RuntimeArtifactSealError(
            f"{expected.label} runtime artifact seal changed"
        )
    return observed


def open_runtime_artifact_seal(
    expected: RuntimeArtifactSeal,
    *,
    required_uid: int,
) -> int:
    """Open and revalidate the exact sealed file for descriptor-pinned use."""

    if expected.required_uid != required_uid:
        raise RuntimeArtifactSealError(
            "runtime artifact seal executor differs"
        )
    path = _lexical_absolute(expected.snapshot.path)
    parent_descriptor, ancestry = _open_parent_tree(
        path,
        required_uid=required_uid,
    )
    descriptor = -1
    verification_parent = -1
    try:
        entry_before = FileIdentity.from_stat(
            os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        )
        descriptor = os.open(
            path.name,
            _file_flags(),
            dir_fd=parent_descriptor,
        )
        opened = FileIdentity.from_stat(os.fstat(descriptor))
        if (
            entry_before != opened
            or opened != expected.snapshot.identity
            or ancestry != expected.ancestry
            or _has_posix_acl(descriptor)
            or _has_forbidden_security_xattr(descriptor)
        ):
            raise RuntimeArtifactSealError(
                "runtime artifact differs while opening its sealed descriptor"
            )
        raw = _read_all(descriptor, maximum=expected.maximum)
        os.lseek(descriptor, 0, os.SEEK_SET)
        opened_after = FileIdentity.from_stat(os.fstat(descriptor))
        entry_after = FileIdentity.from_stat(
            os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        )
        verification_parent, ancestry_after = _open_parent_tree(
            path,
            required_uid=required_uid,
        )
        if (
            opened_after != opened
            or entry_after != opened
            or ancestry_after != ancestry
            or raw != expected.snapshot.raw
            or hashlib.sha256(raw).hexdigest()
            != expected.snapshot.sha256
        ):
            raise RuntimeArtifactSealError(
                "runtime artifact changed while opening its sealed descriptor"
            )
        result = descriptor
        descriptor = -1
        return result
    except RuntimeArtifactSealError:
        raise
    except OSError as error:
        raise RuntimeArtifactSealError(
            "runtime artifact sealed descriptor cannot be opened"
        ) from error
    finally:
        for value in (verification_parent, descriptor, parent_descriptor):
            if value >= 0:
                try:
                    os.close(value)
                except OSError:
                    pass


def _path_has_posix_acl(path: Path) -> bool:
    for attribute in POSIX_ACL_XATTRS:
        try:
            os.getxattr(path, attribute, follow_symlinks=False)
        except OSError as error:
            if error.errno in {
                errno.ENODATA,
                getattr(errno, "ENOATTR", errno.ENODATA),
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                continue
            raise RuntimeArtifactSealError(
                "runtime tree ACL metadata cannot be inspected"
            ) from error
        else:
            return True
    return False


def _path_has_forbidden_security_xattr(path: Path) -> bool:
    for attribute in FORBIDDEN_SECURITY_XATTRS:
        try:
            os.getxattr(path, attribute, follow_symlinks=False)
        except OSError as error:
            if error.errno in {
                errno.ENODATA,
                getattr(errno, "ENOATTR", errno.ENODATA),
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                continue
            raise RuntimeArtifactSealError(
                "runtime tree security metadata cannot be inspected"
            ) from error
        else:
            return True
    return False


def _tree_entry(
    path: Path,
    relative_path: str,
    *,
    required_uid: int,
) -> RuntimeTreeEntry:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeArtifactSealError(
            "runtime tree entry cannot be inspected"
        ) from error
    is_directory = stat.S_ISDIR(metadata.st_mode)
    is_regular = stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeArtifactSealError(
            "runtime tree contains a symbolic link"
        )
    if not is_directory and not is_regular:
        raise RuntimeArtifactSealError(
            "runtime tree contains a special file"
        )
    if metadata.st_uid != required_uid:
        raise RuntimeArtifactSealError(
            "runtime tree entry owner differs"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise RuntimeArtifactSealError(
            "runtime tree entry is group/world writable"
        )
    if is_regular and metadata.st_nlink != 1:
        raise RuntimeArtifactSealError(
            "runtime tree regular file is hard-linked"
        )
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise RuntimeArtifactSealError(
            "runtime tree entry has privilege mode bits"
        )
    if _path_has_posix_acl(path):
        raise RuntimeArtifactSealError(
            "runtime tree entry has a POSIX ACL"
        )
    if _path_has_forbidden_security_xattr(path):
        raise RuntimeArtifactSealError(
            "runtime tree entry has a file capability"
        )
    return RuntimeTreeEntry(
        relative_path=relative_path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        ctime_ns=metadata.st_ctime_ns,
        mode=metadata.st_mode,
        uid=metadata.st_uid,
        gid=metadata.st_gid,
        links=metadata.st_nlink,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _tree_entry_payload(entry: RuntimeTreeEntry) -> bytes:
    fields = (
        entry.relative_path,
        str(entry.device),
        str(entry.inode),
        str(entry.ctime_ns),
        str(entry.mode),
        str(entry.uid),
        str(entry.gid),
        str(entry.links),
        str(entry.size),
        str(entry.mtime_ns),
    )
    encoded = tuple(
        value.encode("utf-8", "surrogateescape") for value in fields
    )
    return b"".join(
        len(value).to_bytes(8, "big") + value
        for value in encoded
    )


def _scan_tree(
    root: Path,
    *,
    label: str,
    required_uid: int,
) -> RuntimeTreeSeal:
    parent_descriptor, ancestry = _open_directory_path(
        root.parent,
        required_uid=required_uid,
    )
    os.close(parent_descriptor)
    entries: list[RuntimeTreeEntry] = []
    pending: list[tuple[Path, str]] = [(root, ".")]
    total_bytes = 0
    while pending:
        path, relative = pending.pop()
        entry = _tree_entry(
            path,
            relative,
            required_uid=required_uid,
        )
        entries.append(entry)
        if len(entries) > MAX_RUNTIME_TREE_ENTRIES:
            raise RuntimeArtifactSealError(
                "runtime tree has too many entries"
            )
        if stat.S_ISREG(entry.mode):
            total_bytes += entry.size
            if total_bytes > MAX_RUNTIME_TREE_TOTAL_BYTES:
                raise RuntimeArtifactSealError(
                    "runtime tree payload is oversized"
                )
            continue
        try:
            children = sorted(
                path.iterdir(),
                key=lambda child: os.fsencode(child.name),
                reverse=True,
            )
        except OSError as error:
            raise RuntimeArtifactSealError(
                "runtime tree directory cannot be enumerated"
            ) from error
        for child in children:
            child_relative = (
                child.name if relative == "." else f"{relative}/{child.name}"
            )
            if len(os.fsencode(child_relative)) > MAX_RELATIVE_PATH_BYTES:
                raise RuntimeArtifactSealError(
                    "runtime tree path is oversized"
                )
            pending.append((child, child_relative))
    verification_descriptor, ancestry_after = _open_directory_path(
        root.parent,
        required_uid=required_uid,
    )
    os.close(verification_descriptor)
    if ancestry != ancestry_after:
        raise RuntimeArtifactSealError(
            "runtime tree ancestry changed while sealing"
        )
    entries.sort(key=lambda entry: os.fsencode(entry.relative_path))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(_tree_entry_payload(entry))
        digest.update(b"\n")
    return RuntimeTreeSeal(
        label=label,
        required_uid=required_uid,
        root=root,
        ancestry=ancestry,
        entries=tuple(entries),
        fingerprint_sha256=digest.hexdigest(),
    )


def capture_runtime_tree_seal(
    root: Path,
    *,
    label: str,
    required_uid: int,
) -> RuntimeTreeSeal:
    """Capture a complete runtime tree without rereading multi-GB payload bytes."""

    if (
        not isinstance(label, str)
        or not label
        or "\x00" in label
        or type(required_uid) is not int
        or required_uid < 0
    ):
        raise RuntimeArtifactSealError(
            "runtime tree seal parameters are invalid"
        )
    selected = _lexical_absolute(root)
    if selected == Path("/"):
        raise RuntimeArtifactSealError(
            "runtime tree root cannot be the filesystem root"
        )
    first = _scan_tree(
        selected,
        label=label,
        required_uid=required_uid,
    )
    second = _scan_tree(
        selected,
        label=label,
        required_uid=required_uid,
    )
    if first != second:
        raise RuntimeArtifactSealError(
            "runtime tree changed while sealing"
        )
    return first


def require_runtime_tree_seal(
    expected: RuntimeTreeSeal,
    *,
    required_uid: int,
) -> RuntimeTreeSeal:
    """Require the exact recursive metadata fingerprint captured at admission."""

    if expected.required_uid != required_uid:
        raise RuntimeArtifactSealError(
            "runtime tree seal executor differs"
        )
    observed = capture_runtime_tree_seal(
        expected.root,
        label=expected.label,
        required_uid=required_uid,
    )
    if observed != expected:
        raise RuntimeArtifactSealError(
            f"{expected.label} runtime tree seal changed"
        )
    return observed
