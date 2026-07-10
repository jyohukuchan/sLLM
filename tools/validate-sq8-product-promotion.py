#!/usr/bin/env python3
"""Validate the immutable SQ8 v0.1 product artifact and thin package copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import sq8_canonical_artifact as canonical  # noqa: E402


SCHEMA_VERSION = "ullm.sq8_product_promotion.v1"
DEFAULT_PRODUCT_ROOT = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_MODEL_ID = "Qwen/Qwen3-14B-FP8"
EXPECTED_MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
EXPECTED_PLAN_COMMIT = "dfc63de"
EXPECTED_ARTIFACT = {
    "schema_version": "sq-fp8-artifact-v0.2",
    "manifest_bytes": 379_114,
    "manifest_sha256": "23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2",
    "content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "selected_pair_count": 280,
    "payload_bytes": 13_213_670_400,
    "file_count": 561,
}
EXPECTED_PACKAGE = {
    "schema_version": "ullm-prototype-manifest-v0.1",
    "manifest_bytes": 91_910,
    "manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "payload_count": 163,
    "file_count": 164,
}


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product_root", nargs="?", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate manifests, paths, sizes, and permissions without hashing payload files.",
    )
    return parser.parse_args()


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant: {value}")


def read_json(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    require_regular_file(path, path.parent)
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValidationError(f"JSON file exceeds {maximum_bytes} bytes: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=duplicate_rejecting_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, root: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValidationError(f"cannot inspect required file {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"required path is not a regular non-symlink file: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(f"required file escapes root {root}: {path}") from error


def safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value.startswith("./")
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValidationError(f"{label} is unsafe: {value}")
    return relative


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def validate_promotion(root: Path) -> dict[str, Any]:
    path = root / "promotion.json"
    promotion = read_json(path)
    exact_keys(
        promotion,
        {"schema_version", "created_at", "plan_commit", "model", "artifact", "package", "copy"},
        "promotion",
    )
    if promotion["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("promotion schema_version mismatch")
    try:
        datetime.fromisoformat(promotion["created_at"])
    except (TypeError, ValueError) as error:
        raise ValidationError("promotion created_at is not ISO-8601") from error
    if promotion["plan_commit"] != EXPECTED_PLAN_COMMIT:
        raise ValidationError("promotion plan_commit mismatch")
    model = promotion["model"]
    if model != {"id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION}:
        raise ValidationError("promotion model identity mismatch")

    artifact = promotion["artifact"]
    package = promotion["package"]
    copy = promotion["copy"]
    if not isinstance(artifact, dict) or not isinstance(package, dict) or not isinstance(copy, dict):
        raise ValidationError("promotion artifact/package/copy entries must be objects")
    expected_artifact = {
        "source": "/tmp/ullm-qwen3-14b-fp8-sq8-canonical-full-v0.2",
        "destination": str((root / "artifact").resolve()),
        **EXPECTED_ARTIFACT,
        "verified": True,
    }
    expected_package = {
        "source": "/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d",
        "destination": str((root / "package").resolve()),
        **EXPECTED_PACKAGE,
        "verified": True,
    }
    if artifact != expected_artifact:
        raise ValidationError("promotion artifact record mismatch")
    if package != expected_package:
        raise ValidationError("promotion package record mismatch")
    if copy != {
        "method": "rsync_archive_streaming",
        "source_and_destination_manifests_byte_identical": True,
        "destination_read_only": True,
    }:
        raise ValidationError("promotion copy record mismatch")
    return promotion


def validate_read_only_tree(root: Path, expected_files: int) -> None:
    file_count = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValidationError(f"read-only product tree contains a symlink: {path}")
        if info.st_mode & 0o222:
            raise ValidationError(f"read-only product entry has write bits: {path}")
        if stat.S_ISREG(info.st_mode):
            file_count += 1
        elif not stat.S_ISDIR(info.st_mode):
            raise ValidationError(f"unsupported product entry type: {path}")
    if file_count != expected_files:
        raise ValidationError(
            f"product file count mismatch for {root}: expected {expected_files} got {file_count}"
        )


def validate_artifact(root: Path, *, full_payloads: bool) -> dict[str, Any]:
    manifest_path = root / "sq_manifest.json"
    require_regular_file(manifest_path, root)
    if manifest_path.stat().st_size != EXPECTED_ARTIFACT["manifest_bytes"]:
        raise ValidationError("artifact manifest byte length mismatch")
    if sha256_file(manifest_path) != EXPECTED_ARTIFACT["manifest_sha256"]:
        raise ValidationError("artifact manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != EXPECTED_ARTIFACT["schema_version"]:
        raise ValidationError("artifact schema mismatch")
    if manifest.get("integrity", {}).get("content_sha256") != EXPECTED_ARTIFACT["content_sha256"]:
        raise ValidationError("artifact content identity mismatch")
    if manifest.get("coverage", {}).get("selected_pair_count") != EXPECTED_ARTIFACT["selected_pair_count"]:
        raise ValidationError("artifact selected pair count mismatch")
    if manifest.get("storage", {}).get("total_payload_bytes") != EXPECTED_ARTIFACT["payload_bytes"]:
        raise ValidationError("artifact payload byte count mismatch")
    if full_payloads:
        try:
            result = canonical.verify_canonical_artifact(root)
        except Exception as error:
            raise ValidationError(f"canonical artifact verification failed: {error}") from error
        if result.get("verified") is not True:
            raise ValidationError("canonical artifact verifier did not report verified=true")
    validate_read_only_tree(root, EXPECTED_ARTIFACT["file_count"])
    return {
        "manifest_sha256": EXPECTED_ARTIFACT["manifest_sha256"],
        "content_sha256": EXPECTED_ARTIFACT["content_sha256"],
        "selected_pair_count": EXPECTED_ARTIFACT["selected_pair_count"],
        "payloads_hashed": full_payloads,
    }


def validate_package(root: Path, *, full_payloads: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    require_regular_file(manifest_path, root)
    if manifest_path.stat().st_size != EXPECTED_PACKAGE["manifest_bytes"]:
        raise ValidationError("package manifest byte length mismatch")
    if sha256_file(manifest_path) != EXPECTED_PACKAGE["manifest_sha256"]:
        raise ValidationError("package manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != EXPECTED_PACKAGE["schema_version"]:
        raise ValidationError("package schema mismatch")
    if manifest.get("tensors") != []:
        raise ValidationError("thin package tensors list must be empty")
    entries = manifest.get("passthrough_tensors")
    if not isinstance(entries, list) or len(entries) != EXPECTED_PACKAGE["payload_count"]:
        raise ValidationError("thin package passthrough count mismatch")
    seen_names: set[str] = set()
    seen_files: set[PurePosixPath] = set()
    payload_bytes = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValidationError(f"package entry {index} is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValidationError(f"invalid or duplicate package tensor name: {name}")
        seen_names.add(name)
        relative = safe_relative_path(entry.get("payload_file"), f"package entry {name} payload_file")
        if relative in seen_files:
            raise ValidationError(f"duplicate package payload file: {relative}")
        seen_files.add(relative)
        path = root.joinpath(*relative.parts)
        require_regular_file(path, root)
        expected_bytes = entry.get("payload_bytes")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise ValidationError(f"invalid package payload byte count: {name}")
        if path.stat().st_size != expected_bytes:
            raise ValidationError(f"package payload byte length mismatch: {name}")
        expected_sha256 = require_sha256(entry.get("payload_sha256"), f"{name}.payload_sha256")
        if full_payloads and sha256_file(path) != expected_sha256:
            raise ValidationError(f"package payload SHA-256 mismatch: {name}")
        payload_bytes += expected_bytes
    validate_read_only_tree(root, EXPECTED_PACKAGE["file_count"])
    return {
        "manifest_sha256": EXPECTED_PACKAGE["manifest_sha256"],
        "payload_count": len(entries),
        "payload_bytes": payload_bytes,
        "payloads_hashed": full_payloads,
    }


def main() -> int:
    args = parse_args()
    root = args.product_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"product root does not exist: {root}")
    full_payloads = not args.metadata_only
    try:
        promotion = validate_promotion(root)
        artifact = validate_artifact(root / "artifact", full_payloads=full_payloads)
        package = validate_package(root / "package", full_payloads=full_payloads)
    except ValidationError as error:
        raise SystemExit(str(error)) from error
    result = {
        "schema_version": SCHEMA_VERSION,
        "product_root": str(root),
        "created_at": promotion["created_at"],
        "model_revision": EXPECTED_MODEL_REVISION,
        "artifact": artifact,
        "package": package,
        "read_only": True,
        "full_payloads": full_payloads,
        "verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
