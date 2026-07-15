"""Strict authorization-lineage manifest validation shared by SQ8 tooling."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "ullm.sq8_authorization_lineage_input.v1"
REFERENCE_SCHEMA = "ullm.sq8_authorization_lineage_ref.v1"
CAPTURE_AUDIT_SCHEMA = (
    "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
)
PROMOTION_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
RUNTIME_AUDIT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
MAX_BYTES = 16 * 1024 * 1024
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
RELATIONS = (
    "implementation_go_eligible_for_fresh_runtime_audit",
    "superseded_capture_implementation_no_go",
    "superseded_capture_implementation_no_go",
    "consumed_actual_failure_latest",
    "consumed_actual_failure_predecessor",
    "superseded_restore_implementation_no_go",
)


class LineageError(ValueError):
    pass


class _DuplicateKey(ValueError):
    pass


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _immutable_bytes(path: Path, label: str) -> tuple[bytes, tuple[int, ...]]:
    if not path.is_absolute() or path != path.resolve():
        raise LineageError(f"{label} path must be canonical absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_BYTES
        ):
            raise LineageError(
                f"{label} must be immutable 0444 single-link bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise LineageError(f"{label} changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LineageError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise LineageError(f"{label} changed while reading")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKey(key)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (_DuplicateKey, UnicodeError, json.JSONDecodeError) as error:
        raise LineageError(f"{label} JSON differs") from error
    if not isinstance(value, dict):
        raise LineageError(f"{label} must be an object")
    return value


def _entry_source(entry: dict[str, Any], index: int) -> dict[str, Any]:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise LineageError("lineage entry path differs")
    path = Path(raw_path)
    raw, _identity = _immutable_bytes(path, f"lineage entry {index}")
    digest = hashlib.sha256(raw).hexdigest()
    if entry.get("sha256") != digest:
        raise LineageError("lineage entry SHA-256 differs")
    source = _json_object(raw, f"lineage entry {index}")
    if source.get("schema_version") != entry.get("schema_version"):
        raise LineageError("lineage entry schema differs")
    return source


def _validate_entry(entry: Any, index: int) -> None:
    if not isinstance(entry, dict) or entry.get("relation") != RELATIONS[index]:
        raise LineageError("lineage entry relation/order differs")
    common = {
        "relation",
        "path",
        "sha256",
        "schema_version",
        "consumed",
        "reusable_as_runtime_authorization",
    }
    relation = RELATIONS[index]
    if index == 0:
        expected = common | {"verdict", "actual"}
    elif index in {1, 2}:
        expected = common | {"verdict", "actual", "reason_codes"}
    elif index in {3, 4}:
        expected = common | {"status", "actual_status", "request_id"}
    else:
        expected = common | {"verdict", "actual", "reason_code"}
    if set(entry) != expected:
        raise LineageError("lineage entry keys differ")
    if entry.get("reusable_as_runtime_authorization") is not False:
        raise LineageError("lineage entry reuse disposition differs")
    if index == 0:
        if entry.get("consumed") is not False:
            raise LineageError("implementation GO consumption state differs")
    elif entry.get("consumed") is not True:
        raise LineageError("historical lineage entry must be consumed")
    source = _entry_source(entry, index)
    if index == 0:
        if (
            entry.get("schema_version") != CAPTURE_AUDIT_SCHEMA
            or entry.get("verdict") != "implementation_ready"
            or entry.get("actual") != "not_executed"
            or source.get("verdict") != entry["verdict"]
            or source.get("actual") != entry["actual"]
            or source.get("authorization", {}).get("eligible_for_fresh_authorization_builder")
            is not True
        ):
            raise LineageError("implementation GO entry differs")
    elif index in {1, 2}:
        if (
            entry.get("schema_version") != CAPTURE_AUDIT_SCHEMA
            or entry.get("verdict") != "implementation_no_go"
            or entry.get("actual") != "not_executed"
            or not isinstance(entry.get("reason_codes"), list)
            or len(entry["reason_codes"]) == 0
            or len(set(entry["reason_codes"])) != len(entry["reason_codes"])
            or not all(
                isinstance(code, str) and code for code in entry["reason_codes"]
            )
            or source.get("verdict") != entry["verdict"]
            or source.get("actual") != entry["actual"]
            or source.get("reason_codes") != entry["reason_codes"]
        ):
            raise LineageError("capture No-Go entry differs")
    elif index in {3, 4}:
        actual = source.get("actual")
        if (
            entry.get("schema_version") != PROMOTION_SCHEMA
            or entry.get("status") != "actual_failed"
            or entry.get("actual_status") != "failed"
            or not isinstance(entry.get("request_id"), str)
            or REQUEST_RE.fullmatch(entry["request_id"]) is None
            or source.get("status") != entry["status"]
            or source.get("request_id") != entry["request_id"]
            or not isinstance(actual, dict)
            or actual.get("status") != entry["actual_status"]
            or actual.get("request_id") != entry["request_id"]
        ):
            raise LineageError("actual failure entry differs")
    elif (
        entry.get("schema_version") != RUNTIME_AUDIT_SCHEMA
        or entry.get("verdict") != "implementation_no_go"
        or entry.get("actual") != "not_executed"
        or entry.get("reason_code") != "restore_retry_terminal_identity_not_fail_closed"
        or source.get("verdict") != entry["verdict"]
        or source.get("actual") != entry["actual"]
        or source.get("reason_code") != entry["reason_code"]
    ):
        raise LineageError("restore No-Go entry differs")
    if relation != entry["relation"]:
        raise LineageError("lineage entry relation differs")


def validate_manifest(
    path: Path, *, expected_source: dict[str, str] | None = None
) -> dict[str, Any]:
    raw, identity = _immutable_bytes(path, "authorization lineage manifest")
    document = _json_object(raw, "authorization lineage manifest")
    if set(document) != {"schema_version", "disposition", "source", "entries"}:
        raise LineageError("authorization lineage manifest keys differ")
    if (
        document.get("schema_version") != MANIFEST_SCHEMA
        or document.get("disposition")
        != "authorization_input_not_yet_runtime_bound"
    ):
        raise LineageError("authorization lineage manifest state differs")
    source = document.get("source")
    if not isinstance(source, dict) or set(source) != {
        "commit",
        "tree_oid",
        "archive_sha256",
    }:
        raise LineageError("authorization lineage source shape differs")
    if expected_source is not None and source != expected_source:
        raise LineageError("authorization lineage source identity differs")
    if (
        not isinstance(source.get("commit"), str)
        or HEX40_RE.fullmatch(source["commit"]) is None
        or not isinstance(source.get("tree_oid"), str)
        or HEX40_RE.fullmatch(source["tree_oid"]) is None
        or not isinstance(source.get("archive_sha256"), str)
        or HEX64_RE.fullmatch(source["archive_sha256"]) is None
    ):
        raise LineageError("authorization lineage source types differ")
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) != len(RELATIONS):
        raise LineageError("authorization lineage entry count differs")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        _validate_entry(entry, index)
        paths.append(entry["path"])
    if len(set(paths)) != len(paths):
        raise LineageError("authorization lineage entry path is duplicated")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries_sha256": canonical_sha(entries),
        "raw": raw,
        "identity": identity,
        "document": document,
    }


def make_reference(validated: dict[str, Any], runtime_path: Path) -> dict[str, str]:
    return {
        "schema_version": REFERENCE_SCHEMA,
        "input_path": validated["path"],
        "runtime_path": str(runtime_path.resolve()),
        "sha256": validated["sha256"],
        "entries_sha256": validated["entries_sha256"],
    }


def validate_reference(
    value: Any, *, expected_runtime_path: Path | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "input_path",
        "runtime_path",
        "sha256",
        "entries_sha256",
    }:
        raise LineageError("authorization lineage reference keys differ")
    if value.get("schema_version") != REFERENCE_SCHEMA:
        raise LineageError("authorization lineage reference schema differs")
    input_path = Path(str(value.get("input_path", "")))
    runtime_path = Path(str(value.get("runtime_path", "")))
    if expected_runtime_path is not None and runtime_path != expected_runtime_path.resolve():
        raise LineageError("authorization lineage runtime path differs")
    validated = validate_manifest(input_path)
    runtime = validate_manifest(runtime_path)
    if (
        value.get("sha256") != validated["sha256"]
        or value.get("entries_sha256") != validated["entries_sha256"]
        or runtime["sha256"] != validated["sha256"]
        or runtime["entries_sha256"] != validated["entries_sha256"]
    ):
        raise LineageError("authorization lineage reference digest differs")
    return value
