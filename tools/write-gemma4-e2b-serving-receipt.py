#!/usr/bin/env python3
"""Write the exact receipt admitted by the Gemma4 E2B served-model profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


RECEIPT_SCHEMA = "ullm.gemma4_e2b_serving_receipt.v1"
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")


class ReceiptError(RuntimeError):
    """Raised when a receipt cannot bind immutable serving inputs."""


def _reject_symlink_components(path: Path, label: str, *, leaf_may_absent: bool) -> None:
    if not path.is_absolute():
        raise ReceiptError(f"{label} must be absolute")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        if part in {"", ".", ".."}:
            raise ReceiptError(f"{label} is not canonical")
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(path.parts[1:]) - 1:
                return
            raise ReceiptError(f"{label} has an absent path component") from None
        except OSError as error:
            raise ReceiptError(f"failed to inspect {label}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ReceiptError(f"{label} traverses a symlink")


def _sha256_file(path: Path, label: str) -> str:
    path = path.absolute()
    _reject_symlink_components(path, label, leaf_may_absent=False)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReceiptError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ReceiptError(f"{label} must be a single-link regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(
    *,
    source_commit: str,
    worker_binary: Path,
    package_manifest: Path,
    tokenizer_config: Path,
    output: Path,
) -> dict[str, Any]:
    if _SHA1.fullmatch(source_commit) is None:
        raise ReceiptError("source commit must be a full lowercase Git SHA-1")
    worker_binary = worker_binary.absolute()
    package_manifest = package_manifest.absolute()
    tokenizer_config = tokenizer_config.absolute()
    output = output.absolute()
    _reject_symlink_components(output, "receipt output", leaf_may_absent=True)
    if output.exists() or output.is_symlink():
        raise ReceiptError("receipt output already exists")
    try:
        parent_metadata = output.parent.lstat()
    except OSError as error:
        raise ReceiptError("receipt output parent is unavailable") from error
    if not stat.S_ISDIR(parent_metadata.st_mode) or parent_metadata.st_mode & 0o022:
        raise ReceiptError("receipt output parent is unsafe")
    worker_sha256 = _sha256_file(worker_binary, "worker binary")
    package_sha256 = _sha256_file(package_manifest, "package manifest")
    _sha256_file(tokenizer_config, "tokenizer config")
    try:
        tokenizer = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptError("tokenizer config is invalid") from error
    template = tokenizer.get("chat_template") if isinstance(tokenizer, dict) else None
    if not isinstance(template, str) or not template:
        raise ReceiptError("tokenizer config has no chat template")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_commit": source_commit,
        "worker_binary_sha256": worker_sha256,
        "package_manifest_sha256": package_sha256,
        "tokenizer_chat_template_sha256": hashlib.sha256(
            template.encode("utf-8")
        ).hexdigest(),
    }
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(
                (json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode(
                    "utf-8"
                )
            )
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(0o444)
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        temporary = None
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker-binary", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = write_receipt(
            source_commit=args.source_commit,
            worker_binary=args.worker_binary,
            package_manifest=args.package_manifest,
            tokenizer_config=args.tokenizer_config,
            output=args.output,
        )
    except Exception as error:
        print(f"Gemma4 E2B receipt write failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
