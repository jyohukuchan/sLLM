#!/usr/bin/env python3
"""Build and seal an SQ8 worker from one clean detached Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


RECEIPT_SCHEMA = "ullm.sq8_worker_build_receipt.v1"
PROVENANCE_SCHEMA = "ullm.sq8_worker_build_provenance.v1"
SEAL_SCHEMA = "ullm.sq8_worker_release_seal.v1"
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_TEXT_BYTES = 1_048_576
SOURCE_INPUTS = (
    "Cargo.lock",
    "Cargo.toml",
    ".cargo/config.toml",
    "crates/ullm-engine/Cargo.toml",
    "crates/ullm-runtime-sys/Cargo.toml",
    "crates/ullm-runtime-sys/build.rs",
    "crates/ullm-engine/src/bin/ullm-sq8-worker.rs",
    "crates/ullm-engine/src/reasoning.rs",
    "crates/ullm-engine/src/served_model.rs",
    "crates/ullm-engine/src/sq8_sampling.rs",
    "crates/ullm-engine/src/sq8_serving_runtime.rs",
    "crates/ullm-engine/src/sq8_worker_backend.rs",
    "crates/ullm-engine/src/sq8_worker_protocol.rs",
    "crates/ullm-engine/src/sq8_worker_runtime.rs",
)
BUILD_ARGUMENTS = (
    "build",
    "--locked",
    "--release",
    "-p",
    "ullm-engine",
    "--bin",
    "ullm-sq8-worker",
    "--features",
    "rocm-ck-gfx1201",
)
BUILD_OVERRIDES = {
    "CARGO_BUILD_JOBS": "1",
    "CARGO_INCREMENTAL": "0",
    "GPU_ARCH": "gfx1201",
    "ROCM_PATH": "/opt/rocm",
    "CUDA_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
    "ULLM_HIP_VISIBLE_DEVICES": "-1",
}
DISALLOWED_BUILD_ENVIRONMENT = {
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
    "CARGO_PROFILE_RELEASE_DEBUG",
    "CARGO_PROFILE_RELEASE_LTO",
    "CARGO_PROFILE_RELEASE_OPT_LEVEL",
    "CARGO_PROFILE_RELEASE_PANIC",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
}


class BuildError(RuntimeError):
    """Raised when a clean, sealed worker build cannot be proven."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def canonical_json(document: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise BuildError("build receipt is not canonicalizable") from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BuildError(f"failed to hash {path.name}") from error
    return digest.hexdigest()


def _run(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    label: str,
) -> str:
    try:
        result = runner(
            list(argv),
            cwd=cwd,
            env=environment,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuildError(f"{label} failed") from error
    if result.returncode != 0:
        raise BuildError(f"{label} failed")
    output = result.stdout.strip()
    if len(output.encode("utf-8")) > MAX_TEXT_BYTES:
        raise BuildError(f"{label} output is oversized")
    return output


def _git(
    runner: CommandRunner, repo_root: Path, *arguments: str, label: str
) -> str:
    return _run(
        runner,
        ("git", *arguments),
        cwd=repo_root,
        label=label,
    )


def _reject_symlink_components(
    path: Path, label: str, *, leaf_may_absent: bool
) -> None:
    if not path.is_absolute():
        raise BuildError(f"{label} must be absolute")
    current = Path(path.anchor)
    components = path.parts[1:]
    for index, component in enumerate(components):
        if component in {"", ".", ".."}:
            raise BuildError(f"{label} is not canonical")
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(components) - 1:
                return
            raise BuildError(f"{label} has an absent path component") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise BuildError(f"{label} traverses a symlink")


def _fresh_path(path: Path, label: str) -> None:
    _reject_symlink_components(path, label, leaf_may_absent=True)
    if path.exists() or path.is_symlink():
        raise BuildError(f"{label} already exists")
    metadata = path.parent.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        raise BuildError(f"{label} parent is unsafe")


def _source_identity(
    repo_root: Path, runner: CommandRunner
) -> tuple[str, str, int, dict[str, dict[str, Any]]]:
    resolved = Path(
        _git(runner, repo_root, "rev-parse", "--show-toplevel", label="Git root")
    ).resolve(strict=True)
    if resolved != repo_root.resolve(strict=True):
        raise BuildError("repository root differs from Git top-level")
    commit = _git(runner, repo_root, "rev-parse", "HEAD", label="Git commit")
    tree = _git(runner, repo_root, "rev-parse", "HEAD^{tree}", label="Git tree")
    if GIT_RE.fullmatch(commit) is None or GIT_RE.fullmatch(tree) is None:
        raise BuildError("Git source identity is not a full object ID")
    status = _git(
        runner,
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        label="Git status",
    )
    if status:
        raise BuildError("source worktree is not clean")
    # `git symbolic-ref -q HEAD` exits 1 for the required detached state, so
    # inspect it separately rather than treating that return code as failure.
    detached_result = runner(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo_root,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    if detached_result.returncode != 1 or detached_result.stdout:
        raise BuildError("source worktree is not detached")
    timestamp_text = _git(
        runner,
        repo_root,
        "show",
        "-s",
        "--format=%ct",
        "HEAD",
        label="Git commit timestamp",
    )
    try:
        source_date_epoch = int(timestamp_text, 10)
    except ValueError as error:
        raise BuildError("Git commit timestamp is invalid") from error
    if source_date_epoch <= 0:
        raise BuildError("Git commit timestamp is invalid")
    inputs: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_INPUTS:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise BuildError(f"build source input is unavailable: {relative}")
        inputs[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return commit, tree, source_date_epoch, inputs


def _tool_version(
    runner: CommandRunner, repo_root: Path, executable: str, *arguments: str
) -> dict[str, str]:
    binary = shutil.which(executable)
    if binary is None:
        raise BuildError(f"required tool is unavailable: {executable}")
    output = _run(
        runner,
        (binary, *arguments),
        cwd=repo_root,
        label=f"{executable} version",
    )
    first_line = output.splitlines()[0] if output else ""
    if not first_line:
        raise BuildError(f"{executable} version is empty")
    return {
        "path": os.fspath(Path(binary).resolve(strict=True)),
        "sha256": sha256_file(Path(binary).resolve(strict=True)),
        "version": first_line,
    }


def _exclusive_write(path: Path, raw: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise BuildError(f"failed to create release member {path.name}") from error
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BuildError(f"release member write made no progress: {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_copy(source: Path, destination: Path, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, mode)
    except OSError as error:
        raise BuildError("failed to create sealed worker") from error
    try:
        os.fchmod(descriptor, mode)
        with source.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise BuildError("sealed worker write made no progress")
                    view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_release(
    repo_root: Path,
    output: Path,
    target_directory: Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    output = output.absolute()
    target_directory = target_directory.absolute()
    _fresh_path(output, "release output")
    _fresh_path(target_directory, "Cargo target directory")
    commit, tree, source_date_epoch, inputs = _source_identity(repo_root, runner)
    for name in DISALLOWED_BUILD_ENVIRONMENT:
        if os.environ.get(name):
            raise BuildError(f"ambient build override is forbidden: {name}")
    cargo = shutil.which("cargo")
    if cargo is None:
        raise BuildError("cargo is unavailable")
    environment = os.environ.copy()
    environment.update(BUILD_OVERRIDES)
    environment["CARGO_TARGET_DIR"] = os.fspath(target_directory)
    environment["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    started_ns = time.time_ns()
    _run(
        runner,
        (cargo, *BUILD_ARGUMENTS),
        cwd=repo_root,
        environment=environment,
        label="SQ8 release build",
    )
    finished_ns = time.time_ns()
    worker_source = target_directory / "release/ullm-sq8-worker"
    if worker_source.is_symlink() or not worker_source.is_file():
        raise BuildError("Cargo did not produce the SQ8 worker")
    source_mode = stat.S_IMODE(worker_source.stat().st_mode)
    if not source_mode & stat.S_IXUSR:
        raise BuildError("built SQ8 worker is not executable")

    toolchain = {
        "cargo": _tool_version(runner, repo_root, "cargo", "--version"),
        "rustc": _tool_version(runner, repo_root, "rustc", "--version"),
        "cxx": _tool_version(runner, repo_root, "c++", "--version"),
        "hipcc": _tool_version(runner, repo_root, "hipcc", "--version"),
    }
    output.mkdir(mode=0o700)
    worker_output = output / "ullm-sq8-worker"
    _exclusive_copy(worker_source, worker_output, 0o555)
    worker_sha256 = sha256_file(worker_output)
    worker_bytes = worker_output.stat().st_size
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source": {
            "repository_root": os.fspath(repo_root),
            "commit": commit,
            "tree": tree,
            "detached": True,
            "worktree_clean": True,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "build": {
            "argv": [cargo, *BUILD_ARGUMENTS],
            "environment": {
                **BUILD_OVERRIDES,
                "CARGO_TARGET_DIR": os.fspath(target_directory),
                "SOURCE_DATE_EPOCH": str(source_date_epoch),
                "RUSTC_WRAPPER": None,
            },
            "result": "success",
        },
        "inputs": [
            {"path": relative, "sha256": inputs[relative]["sha256"]}
            for relative in sorted(inputs, key=lambda value: value.encode("utf-8"))
        ],
        "worker": {
            "path": os.fspath(worker_output),
            "bytes": worker_bytes,
            "mode": "0555",
            "nlink": 1,
            "sha256": worker_sha256,
        },
    }
    receipt_raw = canonical_json(receipt)
    receipt_path = output / "build-receipt.json"
    _exclusive_write(receipt_path, receipt_raw, 0o444)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "source": {
            "repository_root": os.fspath(repo_root),
            "commit": commit,
            "tree": tree,
            "detached": True,
            "tracked_clean": True,
            "untracked_clean": True,
            "inputs": inputs,
        },
        "build": {
            "argv": [cargo, *BUILD_ARGUMENTS],
            "working_directory": os.fspath(repo_root),
            "target_directory": os.fspath(target_directory),
            "environment": receipt["build"]["environment"],
            "ambient_environment_hermetic": False,
            "ambient_compile_overrides_rejected": sorted(
                DISALLOWED_BUILD_ENVIRONMENT
            ),
            "started_unix_ns": started_ns,
            "finished_unix_ns": finished_ns,
            "toolchain": toolchain,
            "result": "success",
        },
        "worker": {
            **receipt["worker"],
            "protocol": "ullm.worker.v2",
            "format_id": "SQ8_0",
            "model_id": "ullm-qwen3-14b-sq8",
        },
    }
    provenance_path = output / "build-provenance.json"
    _exclusive_write(provenance_path, canonical_json(provenance), 0o444)
    readme = (
        "# SQ8_0 v2 worker release\n\n"
        "This directory is a build artifact, not an activation authorization.\n"
        f"Source commit: `{commit}`\n\n"
        f"Worker SHA-256: `{worker_sha256}`\n"
    ).encode("ascii")
    _exclusive_write(output / "README.md", readme, 0o444)
    members = (
        "README.md",
        "build-provenance.json",
        "build-receipt.json",
        "ullm-sq8-worker",
    )
    sums = "".join(f"{sha256_file(output / name)}  {name}\n" for name in members).encode(
        "ascii"
    )
    _exclusive_write(output / "SHA256SUMS", sums, 0o444)
    seal = {
        "schema_version": SEAL_SCHEMA,
        "source_commit": commit,
        "source_tree": tree,
        "worker_sha256": worker_sha256,
        "build_receipt_sha256": sha256_file(receipt_path),
        "build_provenance_sha256": sha256_file(provenance_path),
        "sha256sums_sha256": sha256_bytes(sums),
        "complete": True,
    }
    _exclusive_write(output / "SEALED.json", canonical_json(seal), 0o444)
    _fsync_directory(output)
    output.chmod(0o555)
    _fsync_directory(output.parent)

    if (
        worker_output.stat().st_nlink != 1
        or stat.S_IMODE(worker_output.stat().st_mode) != 0o555
        or sha256_file(worker_output) != worker_sha256
        or any(
            (output / name).stat().st_nlink != 1
            or stat.S_IMODE((output / name).stat().st_mode) != 0o444
            for name in (
                "README.md",
                "build-provenance.json",
                "build-receipt.json",
                "SHA256SUMS",
                "SEALED.json",
            )
        )
    ):
        raise BuildError("sealed release metadata differs after publication")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_release(
            args.repo_root,
            args.output,
            args.target_directory,
        )
    except BuildError:
        print("SQ8 worker release build failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "source_commit": receipt["source"]["commit"],
                "worker_sha256": receipt["worker"]["sha256"],
                "output": os.fspath(args.output.absolute()),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
