#!/usr/bin/env python3
"""Assemble and validate a hash-only generic reasoning release bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release-bundle.py"
MAX_COMPONENT_BYTES = 16 * 1024 * 1024
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_NAMES = (
    "release_evidence",
    "release_validator",
    "browser_evidence",
    "browser_validator",
    "promotion_evidence",
    "promotion_receipt",
)


class BundleError(RuntimeError):
    """Raised when a release bundle cannot be safely assembled."""


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("input JSON contains duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise BundleError("input JSON contains a non-finite number")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BundleError(f"failed to read {label}") from error
    if not raw or len(raw) > MAX_COMPONENT_BYTES:
        raise BundleError(f"{label} exceeds its size bound")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} root is not an object")
    return value


def _hash_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_COMPONENT_BYTES:
                    raise BundleError(f"{label} exceeds its size bound")
                digest.update(chunk)
    except OSError as error:
        raise BundleError(f"failed to hash {label}") from error
    return digest.hexdigest()


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise BundleError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise BundleError(f"{label} is not a lowercase Git commit")
    return value


def _relative_component(path: Path, bundle_root: Path, label: str) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"{label} must be a regular non-symlink file")
    root = bundle_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise BundleError(f"{label} must be below the bundle directory") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise BundleError(f"{label} path is unsafe")
    for parent in relative.parents:
        if parent == Path("."):
            continue
        if (bundle_root / parent).is_symlink():
            raise BundleError(f"{label} path contains a symlink component")
    return relative.as_posix(), _hash_file(path, label)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ullm_generic_reasoning_release_bundle_preparer_validator", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise BundleError("release bundle validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        raise BundleError("release bundle validator could not be loaded") from error
    return module


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise BundleError("output bundle already exists or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "w", encoding="ascii") as destination:
            json.dump(document, destination, ensure_ascii=True, allow_nan=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    release_evidence: Path,
    release_validator: Path,
    browser_evidence: Path,
    browser_validator: Path,
    promotion_evidence: Path,
    promotion_receipt: Path,
    rollback_manifest: Path,
    systemd_unit: Path,
    environment_file: Path,
    output: Path,
    *,
    status: str = "incomplete",
) -> dict[str, Any]:
    if status not in {"incomplete", "complete"}:
        raise BundleError("bundle status is invalid")
    if output.exists() or output.is_symlink():
        raise BundleError("output bundle already exists or is a symlink")
    bundle_root = output.parent
    release = _read_json(release_evidence, "release evidence")
    if release.get("schema_version") != "ullm.generic_reasoning_release_evidence.v1":
        raise BundleError("release evidence schema differs")
    source_commit = _commit(release.get("source_commit"), "source_commit")
    active_promotion_source_commit = _commit(
        release.get("active_promotion_source_commit"),
        "active_promotion_source_commit",
    )
    identity = release.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "manifest_sha256",
        "worker_binary_sha256",
        "tokenizer_sha256",
        "openwebui_image",
    }:
        raise BundleError("release evidence identity fields differ")
    for field in ("manifest_sha256", "worker_binary_sha256", "tokenizer_sha256"):
        _hash(identity[field], f"identity.{field}")
    if not isinstance(identity["openwebui_image"], str) or "@sha256:" not in identity["openwebui_image"]:
        raise BundleError("identity.openwebui_image is invalid")
    artifacts: dict[str, dict[str, str]] = {}
    inputs = {
        "release_evidence": release_evidence,
        "release_validator": release_validator,
        "browser_evidence": browser_evidence,
        "browser_validator": browser_validator,
        "promotion_evidence": promotion_evidence,
        "promotion_receipt": promotion_receipt,
    }
    for name in ARTIFACT_NAMES:
        relative, digest = _relative_component(inputs[name], bundle_root, name)
        artifacts[name] = {"path": relative, "sha256": digest}
    rollback_target = {}
    for field, path in (
        ("manifest_sha256", rollback_manifest),
        ("systemd_unit_sha256", systemd_unit),
        ("environment_sha256", environment_file),
    ):
        rollback_target[field] = _hash_file(path, f"rollback {field}")
    document = {
        "schema_version": "ullm.generic_reasoning_release_bundle.v1",
        "status": status,
        "production_activation_performed": False,
        "source_commit": source_commit,
        "active_promotion_source_commit": active_promotion_source_commit,
        "identity": identity,
        "artifacts": artifacts,
        "rollback_target": rollback_target,
    }
    validator = _load_validator()
    temporary = output.parent / f".{output.name}.validate"
    try:
        _atomic_write(temporary, document)
        report = validator.validate(temporary)
        if status == "complete" and report["gate_eligible"] is not True:
            raise BundleError("complete bundle is not production-gate eligible")
        _atomic_write(output, document)
        temporary.unlink(missing_ok=True)
        return document
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "release-evidence",
        "release-validator",
        "browser-evidence",
        "browser-validator",
        "promotion-evidence",
        "promotion-receipt",
        "rollback-manifest",
        "systemd-unit",
        "environment-file",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", choices=("incomplete", "complete"), default="incomplete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = prepare(
            args.release_evidence,
            args.release_validator,
            args.browser_evidence,
            args.browser_validator,
            args.promotion_evidence,
            args.promotion_receipt,
            args.rollback_manifest,
            args.systemd_unit,
            args.environment_file,
            args.output,
            status=args.status,
        )
    except Exception as error:
        print(f"Generic reasoning release bundle preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "artifact_count": len(document["artifacts"]),
                "status": document["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
