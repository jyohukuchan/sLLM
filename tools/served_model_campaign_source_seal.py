#!/usr/bin/env python3
"""Fail-closed validation for campaign source repositories executed as root."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


GIT_BINARY = "/usr/bin/git"
GIT_COMMAND_PREFIX = (
    GIT_BINARY,
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.useReplaceRefs=false",
)
GIT_ENVIRONMENT = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
MAX_SOURCE_ENTRIES = 250_000
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 4_096
POSIX_ACL_XATTRS = ("system.posix_acl_access", "system.posix_acl_default")


class SourceSealError(RuntimeError):
    """A source repository is mutable by an identity other than the executor."""


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """One pathname and the metadata that makes replacement observable."""

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
class SourceSeal:
    """A bounded, stable fingerprint of one sealed standalone Git clone."""

    root: Path
    required_uid: int
    entries: tuple[SourceEntry, ...]
    fingerprint_sha256: str


def _lexical_absolute(path: Path) -> Path:
    if not isinstance(path, Path):
        raise SourceSealError("campaign source root is not a pathlib path")
    raw = os.fspath(path)
    if (
        not raw
        or "\x00" in raw
        or not path.is_absolute()
        or path.anchor != "/"
        or raw.startswith("//")
        or os.path.normpath(raw) != raw
    ):
        raise SourceSealError("campaign source root is not lexical absolute")
    return path


def _has_posix_acl(path: Path) -> bool:
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
            raise SourceSealError(
                "campaign source ACL metadata cannot be inspected"
            ) from error
        else:
            return True
    return False


def _stat_entry(path: Path, relative_path: str, *, required_uid: int) -> SourceEntry:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SourceSealError("campaign source entry cannot be inspected") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise SourceSealError("campaign source contains a symbolic link")
    is_directory = stat.S_ISDIR(metadata.st_mode)
    is_regular = stat.S_ISREG(metadata.st_mode)
    if not is_directory and not is_regular:
        raise SourceSealError("campaign source contains a special file")
    if metadata.st_uid != required_uid:
        raise SourceSealError("campaign source entry owner differs")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SourceSealError("campaign source entry is group/world writable")
    if is_regular and metadata.st_nlink != 1:
        raise SourceSealError("campaign source regular file is hard-linked")
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise SourceSealError("campaign source entry has privilege mode bits")
    if _has_posix_acl(path):
        raise SourceSealError("campaign source entry has a POSIX ACL")
    return SourceEntry(
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


def _validate_ancestry(root: Path, *, required_uid: int) -> None:
    components: list[Path] = []
    selected = root
    while True:
        components.append(selected)
        if selected.parent == selected:
            break
        selected = selected.parent
    sticky_boundary_seen = False
    for component in reversed(components):
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SourceSealError(
                "campaign source ancestry cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SourceSealError(
                "campaign source ancestry is not symlink-free directories"
            )
        if _has_posix_acl(component):
            raise SourceSealError("campaign source ancestry has a POSIX ACL")
        shared_writable = stat.S_IMODE(metadata.st_mode) & 0o022
        sticky_root = (
            component == Path("/tmp")
            and metadata.st_uid == 0
            and bool(metadata.st_mode & stat.S_ISVTX)
            and bool(shared_writable)
        )
        if sticky_root and not sticky_boundary_seen:
            sticky_boundary_seen = True
            continue
        if shared_writable:
            raise SourceSealError(
                "campaign source ancestry is group/world writable"
            )
        if sticky_boundary_seen:
            if metadata.st_uid != required_uid:
                raise SourceSealError(
                    "campaign source below sticky ancestry has another owner"
                )
        elif metadata.st_uid not in {0, required_uid}:
            raise SourceSealError("campaign source ancestry owner is untrusted")


def _entry_payload(entry: SourceEntry) -> bytes:
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


def _scan(root: Path, *, required_uid: int) -> SourceSeal:
    _validate_ancestry(root, required_uid=required_uid)
    git_path = root / ".git"
    try:
        git_metadata = git_path.lstat()
    except OSError as error:
        raise SourceSealError(
            "campaign source has no internal Git metadata directory"
        ) from error
    if not stat.S_ISDIR(git_metadata.st_mode) or stat.S_ISLNK(git_metadata.st_mode):
        raise SourceSealError(
            "campaign source Git metadata is not an internal directory"
        )
    alternates = git_path / "objects" / "info" / "alternates"
    try:
        alternates.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise SourceSealError("Git alternates path cannot be inspected") from error
    else:
        raise SourceSealError("Git object alternates are forbidden")

    entries: list[SourceEntry] = []
    pending: list[tuple[Path, str]] = [(root, ".")]
    total_bytes = 0
    while pending:
        path, relative = pending.pop()
        entry = _stat_entry(path, relative, required_uid=required_uid)
        entries.append(entry)
        if len(entries) > MAX_SOURCE_ENTRIES:
            raise SourceSealError("campaign source has too many entries")
        if stat.S_ISREG(entry.mode):
            total_bytes += entry.size
            if total_bytes > MAX_SOURCE_TOTAL_BYTES:
                raise SourceSealError("campaign source files are oversized")
            continue
        try:
            children = sorted(
                path.iterdir(),
                key=lambda child: os.fsencode(child.name),
                reverse=True,
            )
        except OSError as error:
            raise SourceSealError(
                "campaign source directory cannot be enumerated"
            ) from error
        for child in children:
            child_relative = (
                child.name if relative == "." else f"{relative}/{child.name}"
            )
            if len(os.fsencode(child_relative)) > MAX_RELATIVE_PATH_BYTES:
                raise SourceSealError("campaign source path is oversized")
            pending.append((child, child_relative))

    entries.sort(key=lambda entry: os.fsencode(entry.relative_path))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(_entry_payload(entry))
        digest.update(b"\n")
    return SourceSeal(
        root=root,
        required_uid=required_uid,
        entries=tuple(entries),
        fingerprint_sha256=digest.hexdigest(),
    )


def capture_source_seal(root: Path, *, required_uid: int) -> SourceSeal:
    """Capture a source only if two complete security scans are identical."""

    if (
        not isinstance(required_uid, int)
        or isinstance(required_uid, bool)
        or required_uid < 0
    ):
        raise SourceSealError("campaign source executor UID is invalid")
    selected = _lexical_absolute(root)
    first = _scan(selected, required_uid=required_uid)
    second = _scan(selected, required_uid=required_uid)
    if first != second:
        raise SourceSealError("campaign source changed while sealing")
    return first


def require_source_seal(
    expected: SourceSeal,
    *,
    required_uid: int,
) -> SourceSeal:
    """Require the exact source metadata fingerprint captured at preflight."""

    if expected.required_uid != required_uid:
        raise SourceSealError("campaign source seal executor differs")
    observed = capture_source_seal(expected.root, required_uid=required_uid)
    if observed != expected:
        raise SourceSealError("campaign source seal changed")
    return observed


def git_argv(arguments: tuple[str, ...] | list[str]) -> list[str]:
    """Build one fixed-path Git invocation with dangerous repo features disabled."""

    return [*GIT_COMMAND_PREFIX, *arguments]


def git_environment() -> dict[str, str]:
    """Return the complete allowlisted environment for Git identity reads."""

    return dict(GIT_ENVIRONMENT)
