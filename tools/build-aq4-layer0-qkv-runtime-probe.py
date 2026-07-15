#!/usr/bin/env python3
"""Build an immutable AQ4 layer-0 runtime probe copy from a clean worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


EXPECTED_COMMIT = "2bcef0d897d43ea1ff397dc558f7e0e179d8a904"
COMMAND_TEXT = "CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe"
BINARY_NAME = "ullm-aq4-layer0-qkv-runtime-probe"
SCHEMA = "ullm.aq4_layer0_qkv_runtime_probe_build_receipt.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular_nlink_one(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must be a regular nlink=1 file: {path}")
    return metadata


def regular_file(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return metadata


def write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, mode)
        try:
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build(source_root: Path, artifact_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    artifact_root = artifact_root.resolve()
    if not (source_root / ".git").exists():
        raise RuntimeError(f"source root is not a Git worktree: {source_root}")
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"source commit differs: {commit}")
    status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise RuntimeError("source worktree is not clean")

    environment = os.environ.copy()
    environment["CARGO_BUILD_JOBS"] = "1"
    completed = subprocess.run(
        ["cargo", "build", "--release", "-p", "ullm-engine", "--bin", BINARY_NAME],
        cwd=source_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"release build failed ({completed.returncode}): "
            f"{completed.stderr.decode(errors='replace')[-4000:]}"
        )
    source_binary = source_root / "target" / "release" / BINARY_NAME
    # Cargo may hard-link a release target into an incremental/dependency location.  The
    # immutable artifact copy below is the nlink=1 boundary; the source only needs to be regular.
    source_metadata = regular_file(source_binary, "release binary")
    artifact_binary = artifact_root / "probe-binary-v0.1" / BINARY_NAME
    if artifact_binary.exists() or artifact_binary.is_symlink():
        raise RuntimeError(f"refusing to overwrite immutable binary: {artifact_binary}")
    artifact_binary.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_binary.with_name(f".{artifact_binary.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary binary already exists: {temporary}")
    try:
        shutil.copyfile(source_binary, temporary)
        os.chmod(temporary, 0o555)
        with temporary.open("rb") as copied:
            os.fsync(copied.fileno())
        os.link(temporary, artifact_binary)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    copied_metadata = regular_nlink_one(artifact_binary, "immutable probe binary")
    binary_sha = sha256(artifact_binary)
    receipt = {
        "schema_version": SCHEMA,
        "status": "ready",
        "source": {
            "worktree": str(source_root),
            "commit": commit,
            "tree_clean": True,
        },
        "build": {
            "command": COMMAND_TEXT,
            "jobs": 1,
            "exit_status": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        },
        "binary": {
            "path": BINARY_NAME,
            "sha256": binary_sha,
            "bytes": copied_metadata.st_size,
            "mode": stat.S_IMODE(copied_metadata.st_mode),
            "nlink": copied_metadata.st_nlink,
            "source_bytes": source_metadata.st_size,
        },
    }
    receipt_path = artifact_binary.parent / "build-receipt.json"
    receipt_payload = (json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
    write_exclusive(receipt_path, receipt_payload, 0o444)
    sums_path = artifact_binary.parent / "SHA256SUMS"
    sums_payload = f"{binary_sha}  {BINARY_NAME}\n{sha256(receipt_path)}  build-receipt.json\n".encode()
    write_exclusive(sums_path, sums_payload, 0o444)
    return {"artifact": str(artifact_root), "binary_sha256": binary_sha, "receipt_sha256": sha256(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.source_root, args.artifact_root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
