#!/usr/bin/env python3
"""Validate and immutably freeze exact served-model manifest bytes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
RESULT_SCHEMA = "ullm.served_model.frozen_manifest.v1"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_MANIFEST_BYTES = 1_048_576
_VALIDATOR_NAME = "_ullm_freeze_served_model_validator"


class FreezeError(RuntimeError):
    """Raised when exact immutable publication cannot be proven."""


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
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


def _reject_symlink_components(
    path: Path,
    label: str,
    *,
    leaf_may_absent: bool,
) -> None:
    if not path.is_absolute():
        raise FreezeError(f"{label} must be absolute")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        if part in {"", ".", ".."}:
            raise FreezeError(f"{label} is not canonical")
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(parts) - 1:
                return
            raise FreezeError(f"{label} has an absent path component") from None
        except OSError as error:
            raise FreezeError(f"failed to inspect {label}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise FreezeError(f"{label} traverses a symlink")


def _stable_read(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label, leaf_may_absent=False)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise FreezeError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FreezeError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_MANIFEST_BYTES
        ):
            raise FreezeError(f"{label} has an invalid file identity")
        raw = bytearray()
        while len(raw) <= MAX_MANIFEST_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, MAX_MANIFEST_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as error:
            raise FreezeError(f"{label} disappeared while being read") from error
        if (
            len(raw) != before.st_size
            or len(raw) > MAX_MANIFEST_BYTES
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named)
        ):
            raise FreezeError(f"{label} changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _load_validator() -> ModuleType:
    existing = sys.modules.get(_VALIDATOR_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_VALIDATOR_NAME, VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise FreezeError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VALIDATOR_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_VALIDATOR_NAME, None)
        raise
    return module


def freeze(source: Path, expected_sha256: str, output: Path) -> dict[str, Any]:
    if HASH_RE.fullmatch(expected_sha256) is None:
        raise FreezeError("expected manifest SHA-256 is invalid")
    if not source.is_absolute() or not output.is_absolute():
        raise FreezeError("source and output paths must be absolute")
    source = source.absolute()
    output = output.absolute()
    if source == output:
        raise FreezeError("source and output paths must differ")
    _reject_symlink_components(output, "frozen manifest output", leaf_may_absent=True)
    if output.exists() or output.is_symlink():
        raise FreezeError("frozen manifest output already exists")
    parent = output.parent
    try:
        parent_metadata = parent.lstat()
        canonical_parent = parent.resolve(strict=True)
    except OSError as error:
        raise FreezeError("frozen manifest output parent is unavailable") from error
    if (
        canonical_parent != parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_mode & stat.S_IWOTH
    ):
        raise FreezeError("frozen manifest output parent is unsafe")
    raw = _stable_read(source, "source served-model manifest")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise FreezeError("source served-model manifest SHA-256 differs")
    try:
        source_summary = _load_validator().validation_summary(source)
    except Exception as error:
        raise FreezeError("source served-model manifest validation failed") from error
    if source_summary.get("manifest_sha256") != digest:
        raise FreezeError("served-model validator manifest identity differs")

    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=parent,
    )
    temporary: Path | None = Path(temporary_value)
    try:
        os.fchmod(descriptor, 0o444)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FreezeError("frozen manifest write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FreezeError("frozen manifest output already exists") from error
        except OSError as error:
            raise FreezeError("frozen manifest publication failed") from error
        temporary.unlink()
        temporary = None
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    observed = _stable_read(output, "frozen served-model manifest")
    metadata = output.lstat()
    if (
        observed != raw
        or hashlib.sha256(observed).hexdigest() != digest
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        raise FreezeError("frozen served-model manifest identity differs")
    try:
        frozen_summary = _load_validator().validation_summary(output)
    except Exception as error:
        raise FreezeError("frozen served-model manifest validation failed") from error
    if frozen_summary != source_summary:
        raise FreezeError("frozen served-model validation summary differs")
    return {
        "schema_version": RESULT_SCHEMA,
        "path": os.fspath(output),
        "bytes": len(raw),
        "sha256": digest,
        "manifest_schema_version": json.loads(raw)["schema_version"],
        "model_id": source_summary["model_id"],
        "format_id": source_summary["format_id"],
        "worker_binary_sha256": source_summary["worker"]["binary_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = freeze(args.source, args.expected_sha256, args.output)
    except Exception:
        print("served-model manifest freeze failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
