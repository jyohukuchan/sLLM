#!/usr/bin/env python3
"""Atomically activate one validated served-model manifest with rollback."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
MAX_MANIFEST_BYTES = 1_048_576
MAX_COMMAND_ARGUMENTS = 128
MAX_COMMAND_ARGUMENT_BYTES = 65_536
MAX_COMMANDS_PER_STAGE = 128
RESULT_SCHEMA = "ullm.served_model.activation.v1"
_VALIDATOR_MODULE_NAME = "_ullm_served_model_activation_validator"


class ActivationError(RuntimeError):
    """Raised when activation cannot finish safely."""


@dataclass(frozen=True, slots=True)
class ActivationResult:
    manifest_sha256: str
    model_id: str
    format_id: str


def load_validator() -> ModuleType:
    """Load the shared preflight implementation without gateway startup imports."""

    existing = sys.modules.get(_VALIDATOR_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _VALIDATOR_MODULE_NAME, VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ActivationError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VALIDATOR_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_VALIDATOR_MODULE_NAME, None)
        raise
    return module


def _reject_symlink_components(path: Path, label: str, *, leaf_may_absent: bool) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(absolute.parts[1:]) - 1:
                return
            raise ActivationError(f"{label} has an absent path component") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ActivationError(f"{label} traverses a symlink")


def _safe_activation_directory(active: Path) -> Path:
    if not active.is_absolute():
        raise ActivationError("active manifest path must be absolute")
    _reject_symlink_components(active, "active manifest", leaf_may_absent=True)
    parent = active.parent
    try:
        metadata = parent.stat()
    except OSError as error:
        raise ActivationError("active manifest directory is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        raise ActivationError("active manifest directory is unsafe")
    return parent.resolve(strict=True)


def _safe_existing_active(active: Path) -> bool:
    try:
        metadata = active.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        raise ActivationError("existing active manifest is unsafe")
    return True


def _read_safe_manifest(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label, leaf_may_absent=False)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActivationError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & stat.S_IWOTH:
            raise ActivationError(f"{label} is unsafe")
        if before.st_size <= 0 or before.st_size > MAX_MANIFEST_BYTES:
            raise ActivationError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ActivationError(f"{label} changed while being read")
        if not raw or len(raw) > MAX_MANIFEST_BYTES or len(raw) != before.st_size:
            raise ActivationError(f"{label} has an invalid size")
        return raw
    finally:
        os.close(descriptor)


def _write_manifest_copy(directory: Path, raw: bytes, *, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=prefix, suffix=".json", dir=directory
    )
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ActivationError("candidate staging write failed")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return path


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot_active(active: Path, directory: Path) -> Path | None:
    if not _safe_existing_active(active):
        return None
    raw = _read_safe_manifest(active, "existing active manifest")
    return _write_manifest_copy(
        directory,
        raw,
        prefix=".served-model.rollback.",
    )


def _restore_active(active: Path, backup: Path | None, directory: Path) -> None:
    if backup is None:
        active.unlink(missing_ok=True)
    else:
        os.replace(backup, active)
    _fsync_directory(directory)


def _parse_command(raw: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("command must be a JSON string array") from error
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_COMMAND_ARGUMENTS
        or any(
            not isinstance(argument, str)
            or not argument
            or "\x00" in argument
            or len(argument.encode("utf-8")) > MAX_COMMAND_ARGUMENT_BYTES
            for argument in value
        )
    ):
        raise argparse.ArgumentTypeError("command must be a bounded JSON string array")
    return tuple(value)


def _run_commands(
    commands: Sequence[Sequence[str]],
    *,
    active: Path,
    summary: dict[str, Any],
    timeout_seconds: float,
    stage: str,
) -> None:
    environment = {
        **os.environ,
        "ULLM_ACTIVE_MANIFEST": os.fspath(active),
        "ULLM_ACTIVE_MANIFEST_SHA256": str(summary["manifest_sha256"]),
        "ULLM_ACTIVE_MODEL_ID": str(summary["model_id"]),
        "ULLM_ACTIVATION_STAGE": stage,
    }
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ActivationError(f"{stage} command failed") from error
        if completed.returncode != 0:
            raise ActivationError(f"{stage} command failed")


def activate(
    candidate: Path,
    active: Path,
    *,
    check_commands: Sequence[Sequence[str]] = (),
    reconcile_commands: Sequence[Sequence[str]] = (),
    final_check_commands: Sequence[Sequence[str]] = (),
    rollback_commands: Sequence[Sequence[str]] = (),
    command_timeout_seconds: float = 60.0,
) -> ActivationResult:
    """Activate a candidate and restore the prior manifest on any later failure."""

    if command_timeout_seconds <= 0:
        raise ActivationError("command timeout must be positive")
    if any(
        len(commands) > MAX_COMMANDS_PER_STAGE
        for commands in (
            check_commands,
            reconcile_commands,
            final_check_commands,
            rollback_commands,
        )
    ):
        raise ActivationError("activation command count exceeds the limit")

    directory = _safe_activation_directory(active)
    normalized_active = directory / active.name
    lock_path = directory / f".{active.name}.activation.lock"
    lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except OSError as error:
        raise ActivationError("activation lock is unavailable") from error
    lock_metadata = os.fstat(lock_descriptor)
    if (
        not stat.S_ISREG(lock_metadata.st_mode)
        or lock_metadata.st_mode & stat.S_IWOTH
    ):
        os.close(lock_descriptor)
        raise ActivationError("activation lock is unsafe")
    staged: Path | None = None
    backup: Path | None = None
    switched = False
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ActivationError("another activation is in progress") from error

        raw = _read_safe_manifest(candidate, "candidate manifest")
        staged = _write_manifest_copy(
            directory,
            raw,
            prefix=".served-model.candidate.",
        )
        try:
            summary = load_validator().validation_summary(staged)
        except Exception as error:
            raise ActivationError("candidate preflight failed") from error

        backup = _snapshot_active(normalized_active, directory)
        # Mark the transaction rollback-capable before the replace so even an
        # asynchronous interruption immediately after it restores the snapshot.
        switched = True
        os.replace(staged, normalized_active)
        staged = None
        _fsync_directory(directory)

        _run_commands(
            check_commands,
            active=normalized_active,
            summary=summary,
            timeout_seconds=command_timeout_seconds,
            stage="check",
        )
        _run_commands(
            reconcile_commands,
            active=normalized_active,
            summary=summary,
            timeout_seconds=command_timeout_seconds,
            stage="reconcile",
        )
        _run_commands(
            final_check_commands,
            active=normalized_active,
            summary=summary,
            timeout_seconds=command_timeout_seconds,
            stage="final-check",
        )
    except BaseException as error:
        if switched:
            try:
                _restore_active(normalized_active, backup, directory)
                backup = None
                _run_commands(
                    rollback_commands,
                    active=normalized_active,
                    summary=summary,
                    timeout_seconds=command_timeout_seconds,
                    stage="rollback",
                )
            except BaseException as rollback_error:
                raise ActivationError("activation and rollback failed") from rollback_error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, ActivationError):
            raise
        raise ActivationError("served-model activation failed") from error
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
        os.close(lock_descriptor)

    return ActivationResult(
        manifest_sha256=str(summary["manifest_sha256"]),
        model_id=str(summary["model_id"]),
        format_id=str(summary["format_id"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--active-manifest", required=True, type=Path)
    parser.add_argument(
        "--check-command-json", action="append", default=[], type=_parse_command
    )
    parser.add_argument(
        "--reconcile-command-json", action="append", default=[], type=_parse_command
    )
    parser.add_argument(
        "--final-check-command-json", action="append", default=[], type=_parse_command
    )
    parser.add_argument(
        "--rollback-command-json", action="append", default=[], type=_parse_command
    )
    parser.add_argument("--command-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = activate(
            args.candidate,
            args.active_manifest,
            check_commands=args.check_command_json,
            reconcile_commands=args.reconcile_command_json,
            final_check_commands=args.final_check_command_json,
            rollback_commands=args.rollback_command_json,
            command_timeout_seconds=args.command_timeout_seconds,
        )
    except Exception:
        # Commands and manifests can contain deployment paths or secrets. Keep
        # CLI failures fixed and intentionally omit caught exception details.
        print("served-model activation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA,
                "activated": True,
                "manifest_sha256": result.manifest_sha256,
                "model_id": result.model_id,
                "format_id": result.format_id,
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
