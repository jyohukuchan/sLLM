#!/usr/bin/env python3
"""Publish an AQ4 resident promotion receipt after fail-closed evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools/generate-served-model.py"
RECEIPT_SCHEMA = "ullm.aq4_resident_promotion.v1"


class ReceiptError(RuntimeError):
    """Raised when a receipt cannot be safely published."""


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ullm_aq4_receipt_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ReceiptError("served-model generator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"failed to read {label}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(profile_path: Path, evidence_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.is_symlink() or output_path.exists():
        raise ReceiptError("output receipt already exists or is a symlink")
    output_path = output_path.resolve()
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReceiptError("evidence must be a regular non-symlink file")
    evidence_path = evidence_path.resolve()
    try:
        relative_evidence = evidence_path.relative_to(output_path.parent)
    except ValueError as error:
        raise ReceiptError("evidence must be inside the receipt directory") from error
    if any(component in ("", ".", "..") for component in relative_evidence.parts):
        raise ReceiptError("evidence path is unsafe")
    evidence = _read_object(evidence_path, "AQ4 promotion evidence")
    source_commit = evidence.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ReceiptError("evidence source commit is invalid")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_commit": source_commit,
        "evidence": {
            "path": os.fspath(relative_evidence),
            "sha256": _sha256_file(evidence_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt: Path | None = None
    temporary_profile: Path | None = None
    try:
        descriptor, raw_receipt = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        temporary_receipt = Path(raw_receipt)
        with os.fdopen(descriptor, "w", encoding="ascii") as destination:
            json.dump(receipt, destination, ensure_ascii=True, allow_nan=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())

        profile = _read_object(profile_path, "AQ4 served-model profile")
        promotion = profile.get("promotion")
        if not isinstance(promotion, dict):
            raise ReceiptError("profile promotion must be an object")
        promotion["receipt"] = os.fspath(temporary_receipt)
        descriptor, raw_profile = tempfile.mkstemp(
            prefix=".aq4-receipt-profile.", suffix=".json", dir=output_path.parent
        )
        temporary_profile = Path(raw_profile)
        with os.fdopen(descriptor, "w", encoding="ascii") as destination:
            json.dump(profile, destination, ensure_ascii=True, allow_nan=False)
            destination.write("\n")
        try:
            _load_generator().materialize(temporary_profile)
        except Exception as error:
            raise ReceiptError(f"AQ4 promotion evidence validation failed: {error}") from error

        temporary_receipt.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary_receipt, output_path)
        temporary_receipt = None
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return receipt
    finally:
        if temporary_profile is not None:
            temporary_profile.unlink(missing_ok=True)
        if temporary_receipt is not None:
            temporary_receipt.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = write_receipt(args.profile, args.evidence, args.output)
    except Exception as error:
        print(f"AQ4 promotion receipt publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
