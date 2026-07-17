#!/usr/bin/env python3
"""Create and verify a detached Phase 6 AQ4 path-oracle binary.

Cargo normally exposes a release executable and its ``deps`` peer as hard
links.  The Phase 6 window deliberately executes a new content copy instead:
the staged file is a regular ``nlink=1`` executable whose SHA-256 is bound to
the clean source-worktree output.  Creation is create-new only; verification
never repairs or rewrites a staging directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ullm.aq4_phase6_path_oracle_binary_staging.v1"
BINARY_NAME = "ullm-aq4-p2-path-oracle"
STAGING_DIRECTORY_MODE = 0o555
STAGING_BINARY_MODE = 0o555
IMMUTABLE_FILE_MODE = 0o444
COPY_CHUNK_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024


class StagingError(RuntimeError):
    """The create-new staging or immutable verification failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagingError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mode_octal(metadata: os.stat_result) -> str:
    return format(stat.S_IMODE(metadata.st_mode), "04o")


def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def metadata_record(metadata: os.stat_result) -> dict[str, int | str]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mode_octal": mode_octal(metadata),
        "nlink": metadata.st_nlink,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
    }


def nofollow_flag() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StagingError("O_NOFOLLOW is unavailable; refusing unsafe staging")
    return os.O_NOFOLLOW


def lstat_regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise StagingError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink: {path}")
    require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file: {path}")
    return metadata


def lstat_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise StagingError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink: {path}")
    require(stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory: {path}")
    return metadata


def assert_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise StagingError(f"could not inspect {label}: {path}: {error}") from error
    raise StagingError(f"refusing to overwrite existing {label}: {path}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise StagingError("short write while creating staging evidence")
        offset += written


def write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    assert_absent(path, "staging evidence file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow_flag()
    descriptor = -1
    try:
        descriptor = os.open(path, flags, mode)
        write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        opened = os.fstat(descriptor)
        require(stat.S_ISREG(opened.st_mode), f"staging evidence changed type while writing: {path}")
        require(opened.st_nlink == 1, f"staging evidence must have nlink=1: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    final = lstat_regular(path, "staging evidence")
    require(mode_octal(final) == format(mode, "04o"), f"staging evidence mode differs: {path}")
    require(final.st_nlink == 1, f"staging evidence must have nlink=1: {path}")


def sha256_regular(
    path: Path,
    label: str,
    *,
    expected_mode: int | None = None,
    require_single_link: bool = False,
    max_bytes: int | None = None,
) -> tuple[str, os.stat_result, bytes | None]:
    before = lstat_regular(path, label)
    if expected_mode is not None:
        require(mode_octal(before) == format(expected_mode, "04o"), f"{label} mode differs: {path}")
    if require_single_link:
        require(before.st_nlink == 1, f"{label} must have nlink=1: {path}")
    if max_bytes is not None:
        require(before.st_size <= max_bytes, f"{label} exceeds bounded size: {path}")
    descriptor = -1
    digest = hashlib.sha256()
    content = bytearray() if max_bytes is not None else None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow_flag())
        opened = os.fstat(descriptor)
        require(identity(before) == identity(opened), f"{label} changed while opening: {path}")
        while True:
            chunk = os.read(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
                require(len(content) <= max_bytes, f"{label} exceeds bounded size while reading: {path}")
        require(identity(opened) == identity(os.fstat(descriptor)), f"{label} changed while reading: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = lstat_regular(path, label)
    require(identity(before) == identity(after), f"{label} changed after reading: {path}")
    return digest.hexdigest(), after, None if content is None else bytes(content)


def copy_source_create_new(source: Path, destination: Path) -> tuple[str, os.stat_result, os.stat_result]:
    source_before = lstat_regular(source, "Cargo path-oracle binary")
    require(source.name == BINARY_NAME, f"unexpected path-oracle basename: {source.name}")
    require(source_before.st_size > 0, "Cargo path-oracle binary was empty")
    assert_absent(destination, "staged path-oracle binary")
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    copied = 0
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow_flag())
        source_opened = os.fstat(source_descriptor)
        require(identity(source_before) == identity(source_opened), "Cargo path-oracle binary changed while opening")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow_flag()
        destination_descriptor = os.open(destination, flags, STAGING_BINARY_MODE)
        while True:
            chunk = os.read(source_descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            write_all(destination_descriptor, chunk)
            copied += len(chunk)
        require(copied == source_before.st_size, "Cargo path-oracle binary size changed while copying")
        require(identity(source_opened) == identity(os.fstat(source_descriptor)), "Cargo path-oracle binary changed while copying")
        os.fsync(destination_descriptor)
        os.fchmod(destination_descriptor, STAGING_BINARY_MODE)
        staged_opened = os.fstat(destination_descriptor)
        require(stat.S_ISREG(staged_opened.st_mode), "staged path-oracle binary changed type while copying")
        require(staged_opened.st_nlink == 1, "staged path-oracle binary must have nlink=1")
        require(mode_octal(staged_opened) == format(STAGING_BINARY_MODE, "04o"), "staged path-oracle binary mode differs")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
    source_after = lstat_regular(source, "Cargo path-oracle binary")
    require(identity(source_before) == identity(source_after), "Cargo path-oracle binary changed after copying")
    staged_after = lstat_regular(destination, "staged path-oracle binary")
    require(staged_after.st_nlink == 1, "staged path-oracle binary must have nlink=1 after copying")
    require(mode_octal(staged_after) == format(STAGING_BINARY_MODE, "04o"), "staged path-oracle binary mode differs after copying")
    return digest.hexdigest(), source_after, staged_after


def validate_commit(value: str) -> None:
    require(len(value) == 40 and all(character in "0123456789abcdef" for character in value), "source commit must be a lowercase 40-character SHA-1")


def verify_stage(stage: Path, expected_source: Path | None, expected_commit: str | None) -> dict[str, Any]:
    stage_metadata = lstat_directory(stage, "path-oracle staging directory")
    require(mode_octal(stage_metadata) == format(STAGING_DIRECTORY_MODE, "04o"), "path-oracle staging directory mode differs")
    expected_names = {BINARY_NAME, "staging-receipt.json", "SHA256SUMS"}
    try:
        names = set(os.listdir(stage))
    except OSError as error:
        raise StagingError(f"could not list path-oracle staging directory: {error}") from error
    require(names == expected_names, f"path-oracle staging members differ: {sorted(names)!r}")

    binary = stage / BINARY_NAME
    binary_sha, binary_metadata, _ = sha256_regular(
        binary,
        "staged path-oracle binary",
        expected_mode=STAGING_BINARY_MODE,
        require_single_link=True,
    )
    receipt = stage / "staging-receipt.json"
    receipt_sha, receipt_metadata, receipt_raw = sha256_regular(
        receipt,
        "path-oracle staging receipt",
        expected_mode=IMMUTABLE_FILE_MODE,
        require_single_link=True,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    assert receipt_raw is not None
    try:
        payload = json.loads(receipt_raw)
    except json.JSONDecodeError as error:
        raise StagingError(f"path-oracle staging receipt was not JSON: {error}") from error
    require(isinstance(payload, dict), "path-oracle staging receipt must be an object")
    require(payload.get("schema_version") == SCHEMA, "path-oracle staging receipt schema differs")
    require(payload.get("status") == "staged", "path-oracle staging receipt status differs")
    staged = payload.get("staged_binary")
    source = payload.get("source")
    require(isinstance(staged, dict) and isinstance(source, dict), "path-oracle staging receipt identities are missing")
    require(staged.get("path") == BINARY_NAME, "path-oracle staging receipt binary path differs")
    require(staged.get("sha256") == binary_sha, "staged path-oracle SHA-256 differs from receipt")
    require(staged.get("bytes") == binary_metadata.st_size, "staged path-oracle byte count differs from receipt")
    require(staged.get("mode_octal") == format(STAGING_BINARY_MODE, "04o"), "staged path-oracle mode differs from receipt")
    require(staged.get("nlink") == 1, "path-oracle staging receipt does not require nlink=1")
    require(source.get("sha256") == binary_sha, "Cargo and staged path-oracle SHA-256 differ")
    if expected_source is not None:
        source_sha, _, _ = sha256_regular(expected_source, "Cargo path-oracle binary")
        require(source.get("path") == str(expected_source), "path-oracle staging source path differs")
        require(source_sha == binary_sha, "Cargo path-oracle SHA-256 differs from staging receipt")
    if expected_commit is not None:
        validate_commit(expected_commit)
        require(payload.get("source_commit") == expected_commit, "path-oracle staging source commit differs")

    sums = stage / "SHA256SUMS"
    _, sums_metadata, sums_raw = sha256_regular(
        sums,
        "path-oracle staging SHA256SUMS",
        expected_mode=IMMUTABLE_FILE_MODE,
        require_single_link=True,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    assert sums_raw is not None
    expected_sums = f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode()
    require(sums_raw == expected_sums, "path-oracle staging SHA256SUMS differs")
    require(identity(stage_metadata) == identity(lstat_directory(stage, "path-oracle staging directory")), "path-oracle staging directory changed while verifying")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "stage_directory": str(stage),
        "source_commit": payload["source_commit"],
        "binary": {"path": str(binary), "sha256": binary_sha, **metadata_record(binary_metadata)},
        "receipt": {"path": str(receipt), "sha256": receipt_sha, **metadata_record(receipt_metadata)},
        "sha256sums": {"path": str(sums), **metadata_record(sums_metadata)},
    }


def stage(source: Path, output: Path, source_commit: str) -> dict[str, Any]:
    validate_commit(source_commit)
    lstat_directory(output.parent, "path-oracle staging parent")
    assert_absent(output, "path-oracle staging directory")
    try:
        output.mkdir(mode=0o700)
    except OSError as error:
        raise StagingError(f"could not create path-oracle staging directory: {output}: {error}") from error
    destination = output / BINARY_NAME
    binary_sha, source_metadata, staged_metadata = copy_source_create_new(source, destination)
    fsync_directory(output)
    receipt = {
        "schema_version": SCHEMA,
        "status": "staged",
        "created_at_utc": utc_now(),
        "source_commit": source_commit,
        "source": {"path": str(source), "sha256": binary_sha, **metadata_record(source_metadata)},
        "staged_binary": {"path": BINARY_NAME, "sha256": binary_sha, **metadata_record(staged_metadata)},
    }
    receipt_path = output / "staging-receipt.json"
    write_exclusive(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(), IMMUTABLE_FILE_MODE)
    receipt_sha, _, _ = sha256_regular(receipt_path, "path-oracle staging receipt", expected_mode=IMMUTABLE_FILE_MODE, require_single_link=True, max_bytes=MAX_RECEIPT_BYTES)
    write_exclusive(
        output / "SHA256SUMS",
        f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode(),
        IMMUTABLE_FILE_MODE,
    )
    os.chmod(output, STAGING_DIRECTORY_MODE)
    fsync_directory(output)
    fsync_directory(output.parent)
    return verify_stage(output, source, source_commit)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="create-new staging directory, or an existing directory with --verify")
    parser.add_argument("--source", type=Path, help="Cargo release path-oracle binary to copy")
    parser.add_argument("--source-commit", help="clean source-worktree commit bound to this binary")
    parser.add_argument("--verify", action="store_true", help="verify an existing immutable staging directory without writing it")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        source = args.source.absolute() if args.source is not None else None
        if args.verify:
            report = verify_stage(args.output, source, args.source_commit)
        else:
            require(source is not None, "--source is required when creating staging")
            require(args.source_commit is not None, "--source-commit is required when creating staging")
            report = stage(source, args.output, args.source_commit)
    except (OSError, StagingError) as error:
        print(f"aq4-phase6-path-oracle-binary-staging: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
