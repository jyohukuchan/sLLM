"""Strict authorization-lineage manifest validation shared by SQ8 tooling."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_V1 = "ullm.sq8_authorization_lineage_input.v1"
REFERENCE_SCHEMA_V1 = "ullm.sq8_authorization_lineage_ref.v1"
MANIFEST_SCHEMA = "ullm.sq8_authorization_lineage_input.v2"
REFERENCE_SCHEMA = "ullm.sq8_authorization_lineage_ref.v2"
CAPTURE_AUDIT_SCHEMA = (
    "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
)
PROMOTION_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
RUNTIME_AUDIT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
MAX_BYTES = 16 * 1024 * 1024
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
V1_RELATIONS = (
    "implementation_go_eligible_for_fresh_runtime_audit",
    "superseded_capture_implementation_no_go",
    "superseded_capture_implementation_no_go",
    "consumed_actual_failure_latest",
    "consumed_actual_failure_predecessor",
    "superseded_restore_implementation_no_go",
)
RELATIONS = frozenset(
    {
        "implementation_ready_current",
        "capture_implementation_no_go",
        "restore_implementation_no_go",
        "actual_failure",
        "historical_implementation_audit",
        "historical_runtime_audit",
    }
)
ENTRY_KEYS = {
    "sequence",
    "relation",
    "path",
    "sha256",
    "schema_version",
    "status",
    "request_id",
    "source_commit",
}


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


def _validate_v1_entry(entry: Any, index: int) -> None:
    if not isinstance(entry, dict) or entry.get("relation") != V1_RELATIONS[index]:
        raise LineageError("lineage entry relation/order differs")
    common = {
        "relation",
        "path",
        "sha256",
        "schema_version",
        "consumed",
        "reusable_as_runtime_authorization",
    }
    relation = V1_RELATIONS[index]
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
            or source.get("authorization", {}).get(
                "eligible_for_fresh_authorization_builder"
            )
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
            or not all(isinstance(code, str) and code for code in entry["reason_codes"])
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


def _validate_source(source: Any, expected_source: dict[str, str] | None) -> None:
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


def _validated_result(
    path: Path,
    raw: bytes,
    identity: tuple[int, ...],
    document: dict[str, Any],
    *,
    authorization_eligible: bool,
    current_implementation_audit: dict[str, str] | None,
) -> dict[str, Any]:
    entries = document["entries"]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries_sha256": canonical_sha(entries),
        "entry_count": len(entries),
        "raw": raw,
        "identity": identity,
        "document": document,
        "authorization_eligible": authorization_eligible,
        "current_implementation_audit": current_implementation_audit,
    }


def _validate_v1_manifest(
    path: Path,
    raw: bytes,
    identity: tuple[int, ...],
    document: dict[str, Any],
    expected_source: dict[str, str] | None,
) -> dict[str, Any]:
    if set(document) != {"schema_version", "disposition", "source", "entries"}:
        raise LineageError("authorization lineage manifest keys differ")
    if (
        document.get("schema_version") != MANIFEST_SCHEMA_V1
        or document.get("disposition") != "authorization_input_not_yet_runtime_bound"
    ):
        raise LineageError("authorization lineage manifest state differs")
    _validate_source(document.get("source"), expected_source)
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) != len(V1_RELATIONS):
        raise LineageError("authorization lineage entry count differs")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        _validate_v1_entry(entry, index)
        paths.append(entry["path"])
    if len(set(paths)) != len(paths):
        raise LineageError("authorization lineage entry path is duplicated")
    return _validated_result(
        path,
        raw,
        identity,
        document,
        authorization_eligible=False,
        current_implementation_audit=None,
    )


def _entry_commit(source: dict[str, Any], schema: str) -> str | None:
    if schema == PROMOTION_SCHEMA:
        value = source.get("source_commit")
    else:
        audited = source.get("audited_source")
        value = audited.get("commit") if isinstance(audited, dict) else None
    return value if isinstance(value, str) else None


def _migrated_v1_entries(validated: dict[str, Any]) -> list[dict[str, Any]]:
    relations = (
        "historical_implementation_audit",
        "capture_implementation_no_go",
        "capture_implementation_no_go",
        "actual_failure",
        "actual_failure",
        "restore_implementation_no_go",
    )
    migrated: list[dict[str, Any]] = []
    for sequence, (entry, relation) in enumerate(
        zip(validated["document"]["entries"], relations, strict=True)
    ):
        source = _entry_source(entry, sequence)
        schema = entry["schema_version"]
        status = entry.get("status", entry.get("verdict"))
        request_id = entry.get("request_id")
        if request_id is None:
            candidate = source.get("fixed_request_id")
            request_id = candidate if isinstance(candidate, str) else None
        source_commit = _entry_commit(source, schema)
        if (
            status
            not in {
                "implementation_ready",
                "implementation_no_go",
                "actual_failed",
            }
            or request_id is not None
            and REQUEST_RE.fullmatch(request_id) is None
            or source_commit is None
            or HEX40_RE.fullmatch(source_commit) is None
        ):
            raise LineageError("v1 lineage entry cannot be canonically migrated")
        migrated.append(
            {
                "sequence": sequence,
                "relation": relation,
                "path": entry["path"],
                "sha256": entry["sha256"],
                "schema_version": schema,
                "status": status,
                "request_id": request_id,
                "source_commit": source_commit,
            }
        )
    return migrated


def _validate_v2_entry(entry: Any, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        raise LineageError("lineage v2 entry keys differ")
    if entry.get("sequence") != index:
        raise LineageError("lineage v2 entries are not canonically sequenced")
    relation = entry.get("relation")
    if relation not in RELATIONS:
        raise LineageError("lineage v2 entry relation differs")
    if (
        not isinstance(entry.get("sha256"), str)
        or HEX64_RE.fullmatch(entry["sha256"]) is None
        or not isinstance(entry.get("source_commit"), str)
        or HEX40_RE.fullmatch(entry["source_commit"]) is None
        or not isinstance(entry.get("schema_version"), str)
        or entry.get("request_id") is not None
        and (
            not isinstance(entry["request_id"], str)
            or REQUEST_RE.fullmatch(entry["request_id"]) is None
        )
    ):
        raise LineageError("lineage v2 entry types differ")
    source = _entry_source(entry, index)
    schema = entry["schema_version"]
    status = entry.get("status")
    request_id = entry.get("request_id")
    source_commit = _entry_commit(source, schema)
    if source_commit != entry["source_commit"]:
        raise LineageError("lineage v2 entry source commit differs")

    if relation == "actual_failure":
        actual = source.get("actual")
        if (
            schema != PROMOTION_SCHEMA
            or status != "actual_failed"
            or request_id is None
            or source.get("status") != status
            or source.get("request_id") != request_id
            or not isinstance(actual, dict)
            or actual.get("status") != "failed"
            or actual.get("request_id") != request_id
        ):
            raise LineageError("lineage v2 actual failure entry differs")
        return source

    if schema not in {CAPTURE_AUDIT_SCHEMA, RUNTIME_AUDIT_SCHEMA}:
        raise LineageError("lineage v2 audit schema differs")
    if request_id is not None and source.get("fixed_request_id") != request_id:
        raise LineageError("lineage v2 audit request differs")
    if source.get("verdict") != status or source.get("actual") != "not_executed":
        raise LineageError("lineage v2 audit status differs")
    if relation == "implementation_ready_current":
        authorization = source.get("authorization")
        eligible = (
            isinstance(authorization, dict)
            and authorization.get("eligible_for_fresh_authorization_builder") is True
        )
        if status != "implementation_ready" or (
            schema == CAPTURE_AUDIT_SCHEMA and not eligible
        ):
            raise LineageError("lineage v2 current implementation GO differs")
    elif relation == "capture_implementation_no_go":
        if schema != CAPTURE_AUDIT_SCHEMA or status != "implementation_no_go":
            raise LineageError("lineage v2 capture No-Go differs")
    elif relation == "restore_implementation_no_go":
        if (
            schema != RUNTIME_AUDIT_SCHEMA
            or status != "implementation_no_go"
            or source.get("reason_code")
            != "restore_retry_terminal_identity_not_fail_closed"
        ):
            raise LineageError("lineage v2 restore No-Go differs")
    elif relation == "historical_implementation_audit":
        if schema != CAPTURE_AUDIT_SCHEMA or status not in {
            "implementation_ready",
            "implementation_no_go",
        }:
            raise LineageError("lineage v2 historical implementation audit differs")
    elif relation == "historical_runtime_audit" and (
        schema != RUNTIME_AUDIT_SCHEMA
        or status not in {"implementation_ready", "implementation_no_go"}
    ):
        raise LineageError("lineage v2 historical runtime audit differs")
    return source


def _validate_v2_manifest(
    path: Path,
    raw: bytes,
    identity: tuple[int, ...],
    document: dict[str, Any],
    expected_source: dict[str, str] | None,
    expected_current_implementation_audit: dict[str, str] | None,
    seen: frozenset[Path],
) -> dict[str, Any]:
    if set(document) != {
        "schema_version",
        "disposition",
        "source",
        "predecessor",
        "entries",
    }:
        raise LineageError("authorization lineage v2 manifest keys differ")
    if document.get("disposition") != "authorization_input_not_yet_runtime_bound":
        raise LineageError("authorization lineage manifest state differs")
    _validate_source(document.get("source"), expected_source)
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise LineageError("authorization lineage v2 entries differ")
    paths: set[str] = set()
    digests: set[str] = set()
    current: list[tuple[int, str, dict[str, str]]] = []
    current_sources: set[str] = set()
    counts = {
        "capture_implementation_no_go": 0,
        "restore_implementation_no_go": 0,
        "actual_failure": 0,
    }
    for index, entry in enumerate(entries):
        _validate_v2_entry(entry, index)
        if entry["path"] in paths or entry["sha256"] in digests:
            raise LineageError("authorization lineage v2 entry is duplicated")
        paths.add(entry["path"])
        digests.add(entry["sha256"])
        relation = entry["relation"]
        if relation == "implementation_ready_current":
            if entry["source_commit"] in current_sources:
                raise LineageError("current implementation GO source is duplicated")
            current_sources.add(entry["source_commit"])
            current.append(
                (
                    index,
                    entry["source_commit"],
                    {"path": entry["path"], "sha256": entry["sha256"]},
                )
            )
        if relation in counts:
            counts[relation] += 1
    if not current:
        raise LineageError("at least one current implementation GO is required")
    current_index, current_source, current_identity = current[-1]
    if current_index != len(entries) - 1:
        raise LineageError("latest current implementation GO must be the final entry")
    if current_source != document["source"]["commit"]:
        raise LineageError("current implementation GO source differs")
    if (
        counts["capture_implementation_no_go"] < 2
        or counts["restore_implementation_no_go"] < 1
        or counts["actual_failure"] < 3
    ):
        raise LineageError("authorization lineage v2 minimum history differs")
    if (
        expected_current_implementation_audit is not None
        and current_identity != expected_current_implementation_audit
    ):
        raise LineageError("current implementation GO receipt binding differs")

    predecessor = document.get("predecessor")
    if not isinstance(predecessor, dict) or predecessor.get("schema_version") not in {
        MANIFEST_SCHEMA_V1,
        MANIFEST_SCHEMA,
    }:
        raise LineageError("authorization lineage predecessor shape differs")
    predecessor_schema = predecessor["schema_version"]
    if predecessor_schema == MANIFEST_SCHEMA_V1:
        if set(predecessor) != {
            "schema_version",
            "path",
            "sha256",
            "migrated_prefix_sha256",
            "migrated_prefix_count",
        }:
            raise LineageError("authorization lineage predecessor shape differs")
        predecessor_path = Path(str(predecessor.get("path", "")))
        if predecessor_path in seen:
            raise LineageError("authorization lineage predecessor cycle differs")
        previous = validate_manifest(
            predecessor_path,
            _seen=seen | {path},
        )
        if previous["document"].get("schema_version") != MANIFEST_SCHEMA_V1:
            raise LineageError("authorization lineage migration predecessor differs")
        migrated = _migrated_v1_entries(previous)
        latest_failure = _entry_source(entries[6], 6)
        previous_source = previous["document"]["source"]
        if (
            predecessor.get("sha256") != previous["sha256"]
            or predecessor.get("migrated_prefix_sha256") != canonical_sha(migrated)
            or predecessor.get("migrated_prefix_count") != len(migrated)
            or len(entries) != len(migrated) + 2
            or entries[: len(migrated)] != migrated
            or entries[6]["relation"] != "actual_failure"
            or entries[6]["source_commit"] != previous_source["commit"]
            or latest_failure.get("source_provenance")
            != {
                "tree_sha256": previous_source["tree_oid"],
                "archive_sha256": previous_source["archive_sha256"],
            }
            or entries[7]["relation"] != "implementation_ready_current"
        ):
            raise LineageError("authorization lineage v1 migration differs")
    else:
        if set(predecessor) != {
            "schema_version",
            "path",
            "sha256",
            "entries_sha256",
            "entry_count",
        }:
            raise LineageError("authorization lineage predecessor shape differs")
        predecessor_path = Path(str(predecessor.get("path", "")))
        if predecessor_path in seen:
            raise LineageError("authorization lineage predecessor cycle differs")
        previous = validate_manifest(predecessor_path, _seen=seen | {path})
        if previous["document"].get("schema_version") != MANIFEST_SCHEMA:
            raise LineageError("authorization lineage predecessor must be v2")
        if (
            predecessor.get("sha256") != previous["sha256"]
            or predecessor.get("entries_sha256") != previous["entries_sha256"]
            or predecessor.get("entry_count") != previous["entry_count"]
            or len(entries) != previous["entry_count"] + 2
            or entries[: previous["entry_count"]] != previous["document"]["entries"]
            or entries[previous["entry_count"]]["relation"] != "actual_failure"
            or entries[previous["entry_count"]]["source_commit"]
            != previous["document"]["source"]["commit"]
            or entries[previous["entry_count"] + 1]["relation"]
            != "implementation_ready_current"
        ):
            raise LineageError("authorization lineage is not append-only")

    return _validated_result(
        path,
        raw,
        identity,
        document,
        authorization_eligible=True,
        current_implementation_audit=current_identity,
    )


def validate_manifest(
    path: Path,
    *,
    expected_source: dict[str, str] | None = None,
    expected_current_implementation_audit: dict[str, str] | None = None,
    _seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    raw, identity = _immutable_bytes(path, "authorization lineage manifest")
    document = _json_object(raw, "authorization lineage manifest")
    schema = document.get("schema_version")
    if schema == MANIFEST_SCHEMA_V1:
        if expected_current_implementation_audit is not None:
            raise LineageError("v1 lineage cannot bind a current implementation GO")
        return _validate_v1_manifest(path, raw, identity, document, expected_source)
    if schema == MANIFEST_SCHEMA:
        return _validate_v2_manifest(
            path,
            raw,
            identity,
            document,
            expected_source,
            expected_current_implementation_audit,
            _seen,
        )
    raise LineageError("authorization lineage manifest schema differs")


def make_reference(validated: dict[str, Any], runtime_path: Path) -> dict[str, Any]:
    if validated["document"]["schema_version"] == MANIFEST_SCHEMA_V1:
        return {
            "schema_version": REFERENCE_SCHEMA_V1,
            "input_path": validated["path"],
            "runtime_path": str(runtime_path.resolve()),
            "sha256": validated["sha256"],
            "entries_sha256": validated["entries_sha256"],
        }
    return {
        "schema_version": REFERENCE_SCHEMA,
        "input_path": validated["path"],
        "runtime_path": str(runtime_path.resolve()),
        "sha256": validated["sha256"],
        "entries_sha256": validated["entries_sha256"],
        "entry_count": validated["entry_count"],
        "current_implementation_audit": validated["current_implementation_audit"],
    }


def validate_reference(
    value: Any, *, expected_runtime_path: Path | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LineageError("authorization lineage reference keys differ")
    schema = value.get("schema_version")
    v1_keys = {
        "schema_version",
        "input_path",
        "runtime_path",
        "sha256",
        "entries_sha256",
    }
    v2_keys = v1_keys | {"entry_count", "current_implementation_audit"}
    if (schema == REFERENCE_SCHEMA_V1 and set(value) != v1_keys) or (
        schema == REFERENCE_SCHEMA and set(value) != v2_keys
    ):
        raise LineageError("authorization lineage reference keys differ")
    if schema not in {REFERENCE_SCHEMA_V1, REFERENCE_SCHEMA}:
        raise LineageError("authorization lineage reference schema differs")
    input_path = Path(str(value.get("input_path", "")))
    runtime_path = Path(str(value.get("runtime_path", "")))
    if (
        expected_runtime_path is not None
        and runtime_path != expected_runtime_path.resolve()
    ):
        raise LineageError("authorization lineage runtime path differs")
    validated = validate_manifest(input_path)
    runtime = validate_manifest(runtime_path)
    expected_manifest_schema = (
        MANIFEST_SCHEMA_V1 if schema == REFERENCE_SCHEMA_V1 else MANIFEST_SCHEMA
    )
    if (
        validated["document"]["schema_version"] != expected_manifest_schema
        or runtime["document"]["schema_version"] != expected_manifest_schema
        or value.get("sha256") != validated["sha256"]
        or value.get("entries_sha256") != validated["entries_sha256"]
        or runtime["sha256"] != validated["sha256"]
        or runtime["entries_sha256"] != validated["entries_sha256"]
    ):
        raise LineageError("authorization lineage reference digest differs")
    if schema == REFERENCE_SCHEMA and (
        value.get("entry_count") != validated["entry_count"]
        or value.get("current_implementation_audit")
        != validated["current_implementation_audit"]
    ):
        raise LineageError("authorization lineage v2 reference metadata differs")
    return value
