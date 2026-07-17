#!/usr/bin/env python3
"""Create and verify the detached AQ4 Phase 3c trace-binary staging copy.

Cargo normally exposes a release binary as a hard link to its ``deps`` copy.
The Phase 3c trace deliberately rejects such a binary because its own identity
contract requires a regular ``nlink=1`` executable.  This tool makes the
required boundary explicit: it copies bytes into a create-new evidence root,
records the source and staged SHA-256 values, and seals the staged executable
as mode 0555 with exactly one link.  It never links, renames, or overwrites a
previous staging path.
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


SCHEMA = "ullm.aq4_phase3c_trace_binary_staging.v1"
BINARY_NAME = "ullm-aq4-differential-trace"
STAGING_DIRECTORY_MODE = 0o555
STAGING_BINARY_MODE = 0o555
IMMUTABLE_FILE_MODE = 0o444
COPY_CHUNK_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024


class StagingError(RuntimeError):
    """The create-new staging or its immutable verification failed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagingError(message)


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
    try:
        return os.O_NOFOLLOW
    except AttributeError as error:  # Linux is required for the Phase 3c contract.
        raise StagingError("O_NOFOLLOW is unavailable; refusing unsafe staging") from error


def regular_lstat(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise StagingError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink: {path}")
    require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file: {path}")
    return metadata


def directory_lstat(path: Path, label: str) -> os.stat_result:
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
        require(stat.S_ISREG(opened.st_mode), f"staging evidence file changed type while writing: {path}")
        require(opened.st_nlink == 1, f"staging evidence file must have nlink=1: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    final = regular_lstat(path, "staging evidence file")
    require(stat.S_IMODE(final.st_mode) == mode, f"staging evidence file mode differs: {path}")
    require(final.st_nlink == 1, f"staging evidence file must have nlink=1: {path}")


def read_regular_sha256(path: Path, label: str, *, expected_mode: int | None = None, max_bytes: int | None = None) -> tuple[str, os.stat_result, bytes | None]:
    before = regular_lstat(path, label)
    if expected_mode is not None:
        require(mode_octal(before) == format(expected_mode, "04o"), f"{label} mode differs: {path}")
    require(before.st_nlink == 1, f"{label} must have nlink=1: {path}")
    if max_bytes is not None:
        require(before.st_size <= max_bytes, f"{label} exceeds bounded size: {path}")
    descriptor = -1
    content = bytearray() if max_bytes is not None else None
    digest = hashlib.sha256()
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
        after_open = os.fstat(descriptor)
        require(identity(opened) == identity(after_open), f"{label} changed while reading: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after_path = regular_lstat(path, label)
    require(identity(before) == identity(after_path), f"{label} changed after reading: {path}")
    return digest.hexdigest(), after_path, None if content is None else bytes(content)


def copy_source_create_new(source: Path, destination: Path) -> tuple[str, os.stat_result, os.stat_result]:
    source_before = regular_lstat(source, "Cargo trace binary")
    require(source.name == BINARY_NAME, f"unexpected trace binary basename: {source.name}")
    require(source_before.st_size > 0, "Cargo trace binary was empty")
    assert_absent(destination, "staged trace binary")
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow_flag())
        source_opened = os.fstat(source_descriptor)
        require(identity(source_before) == identity(source_opened), "Cargo trace binary changed while opening")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | nofollow_flag()
        destination_descriptor = os.open(destination, flags, STAGING_BINARY_MODE)
        while True:
            chunk = os.read(source_descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            write_all(destination_descriptor, chunk)
            total += len(chunk)
        require(total == source_before.st_size, "Cargo trace binary size changed while copying")
        source_after = os.fstat(source_descriptor)
        require(identity(source_opened) == identity(source_after), "Cargo trace binary changed while copying")
        os.fsync(destination_descriptor)
        os.fchmod(destination_descriptor, STAGING_BINARY_MODE)
        destination_opened = os.fstat(destination_descriptor)
        require(stat.S_ISREG(destination_opened.st_mode), "staged trace binary changed type while copying")
        require(destination_opened.st_nlink == 1, "staged trace binary must have nlink=1")
        require(stat.S_IMODE(destination_opened.st_mode) == STAGING_BINARY_MODE, "staged trace binary mode differs")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
    source_final = regular_lstat(source, "Cargo trace binary")
    require(identity(source_before) == identity(source_final), "Cargo trace binary changed after copying")
    destination_final = regular_lstat(destination, "staged trace binary")
    require(stat.S_IMODE(destination_final.st_mode) == STAGING_BINARY_MODE, "staged trace binary mode differs after copying")
    require(destination_final.st_nlink == 1, "staged trace binary must have nlink=1 after copying")
    return digest.hexdigest(), source_final, destination_final


def verify_stage(stage: Path, expected_source: Path | None, expected_commit: str | None) -> dict[str, Any]:
    stage_metadata = directory_lstat(stage, "trace-binary staging directory")
    require(mode_octal(stage_metadata) == format(STAGING_DIRECTORY_MODE, "04o"), "trace-binary staging directory mode differs")
    expected_names = {BINARY_NAME, "staging-receipt.json", "SHA256SUMS"}
    try:
        actual_names = set(os.listdir(stage))
    except OSError as error:
        raise StagingError(f"could not list trace-binary staging directory: {error}") from error
    require(actual_names == expected_names, f"trace-binary staging members differ: {sorted(actual_names)!r}")

    binary = stage / BINARY_NAME
    binary_sha, binary_metadata, _ = read_regular_sha256(binary, "staged trace binary", expected_mode=STAGING_BINARY_MODE)
    receipt = stage / "staging-receipt.json"
    receipt_sha, receipt_metadata, receipt_raw = read_regular_sha256(
        receipt,
        "trace-binary staging receipt",
        expected_mode=IMMUTABLE_FILE_MODE,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    assert receipt_raw is not None
    try:
        receipt_payload = json.loads(receipt_raw)
    except json.JSONDecodeError as error:
        raise StagingError(f"trace-binary staging receipt was not JSON: {error}") from error
    require(isinstance(receipt_payload, dict), "trace-binary staging receipt must be an object")
    require(receipt_payload.get("schema_version") == SCHEMA, "trace-binary staging receipt schema differs")
    require(receipt_payload.get("status") == "staged", "trace-binary staging receipt status differs")
    staged = receipt_payload.get("staged_binary")
    source = receipt_payload.get("source")
    require(isinstance(staged, dict) and isinstance(source, dict), "trace-binary staging receipt identities are missing")
    require(staged.get("path") == BINARY_NAME, "trace-binary staging receipt binary path differs")
    require(staged.get("sha256") == binary_sha, "trace-binary staging SHA-256 differs from receipt")
    require(staged.get("bytes") == binary_metadata.st_size, "trace-binary staging byte count differs from receipt")
    require(staged.get("mode_octal") == format(STAGING_BINARY_MODE, "04o"), "trace-binary staging mode differs from receipt")
    require(staged.get("nlink") == 1, "trace-binary staging receipt does not require nlink=1")
    require(source.get("sha256") == binary_sha, "Cargo and staged trace SHA-256 differ")
    if expected_source is not None:
        require(source.get("path") == str(expected_source), "trace-binary staging source path differs")
    if expected_commit is not None:
        require(receipt_payload.get("trace_tooling_commit") == expected_commit, "trace-binary staging tooling commit differs")

    sums = stage / "SHA256SUMS"
    _, sums_metadata, sums_raw = read_regular_sha256(
        sums,
        "trace-binary staging SHA256SUMS",
        expected_mode=IMMUTABLE_FILE_MODE,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    assert sums_raw is not None
    expected_sums = f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode()
    require(sums_raw == expected_sums, "trace-binary staging SHA256SUMS differs")
    stage_after = directory_lstat(stage, "trace-binary staging directory")
    require(identity(stage_metadata) == identity(stage_after), "trace-binary staging directory changed while verifying")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "stage_directory": str(stage),
        "trace_tooling_commit": receipt_payload["trace_tooling_commit"],
        "binary": {"path": str(binary), "sha256": binary_sha, **metadata_record(binary_metadata)},
        "receipt": {"path": str(receipt), "sha256": receipt_sha, **metadata_record(receipt_metadata)},
        "sha256sums": {"path": str(sums), **metadata_record(sums_metadata)},
    }


def stage(source: Path, output: Path, trace_tooling_commit: str) -> dict[str, Any]:
    require(len(trace_tooling_commit) == 40 and all(character in "0123456789abcdef" for character in trace_tooling_commit), "trace tooling commit must be a lowercase 40-character SHA-1")
    directory_lstat(output.parent, "trace-binary staging parent")
    assert_absent(output, "trace-binary staging directory")
    try:
        output.mkdir(mode=0o700)
    except OSError as error:
        raise StagingError(f"could not create trace-binary staging directory: {output}: {error}") from error
    destination = output / BINARY_NAME
    binary_sha, source_metadata, staged_metadata = copy_source_create_new(source, destination)
    fsync_directory(output)
    receipt = {
        "schema_version": SCHEMA,
        "status": "staged",
        "created_at_utc": utc_now(),
        "trace_tooling_commit": trace_tooling_commit,
        "source": {"path": str(source), "sha256": binary_sha, **metadata_record(source_metadata)},
        "staged_binary": {"path": BINARY_NAME, "sha256": binary_sha, **metadata_record(staged_metadata)},
    }
    receipt_path = output / "staging-receipt.json"
    write_exclusive(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(), IMMUTABLE_FILE_MODE)
    receipt_sha, _, _ = read_regular_sha256(receipt_path, "trace-binary staging receipt", expected_mode=IMMUTABLE_FILE_MODE, max_bytes=MAX_RECEIPT_BYTES)
    sums_path = output / "SHA256SUMS"
    write_exclusive(
        sums_path,
        f"{binary_sha}  {BINARY_NAME}\n{receipt_sha}  staging-receipt.json\n".encode(),
        IMMUTABLE_FILE_MODE,
    )
    os.chmod(output, STAGING_DIRECTORY_MODE)
    fsync_directory(output)
    fsync_directory(output.parent)
    return verify_stage(output, source, trace_tooling_commit)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="create-new staging directory, or an existing directory with --verify")
    parser.add_argument("--source", type=Path, help="Cargo release trace binary to copy")
    parser.add_argument("--trace-tooling-commit", help="fixed 40-character trace tooling commit")
    parser.add_argument("--verify", action="store_true", help="verify an existing immutable staging directory without writing it")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.verify:
            if args.source is not None:
                args.source = args.source.absolute()
            report = verify_stage(args.output, args.source, args.trace_tooling_commit)
        else:
            require(args.source is not None, "--source is required when creating staging")
            require(args.trace_tooling_commit is not None, "--trace-tooling-commit is required when creating staging")
            report = stage(args.source.absolute(), args.output, args.trace_tooling_commit)
    except (OSError, StagingError) as error:
        print(f"aq4-phase3c-trace-binary-staging: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
