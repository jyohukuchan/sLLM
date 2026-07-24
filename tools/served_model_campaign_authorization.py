#!/usr/bin/env python3
"""Strict authorization and one-shot claim primitives for a v2 campaign window."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_aq4_restoration_proof as restoration_proof


AUTHORIZATION_SCHEMA = (
    "ullm.served_model.v2_cross_model_campaign_authorization.v2"
)
CLAIM_SCHEMA = "ullm.served_model.v2_cross_model_campaign_claim.v2"
OUTCOME_SCHEMA = "ullm.served_model.v2_cross_model_campaign_outcome.v2"
RECOVERY_SCHEMA = "ullm.served_model.v2_cross_model_campaign_recovery.v2"
FIXED_CLAIM_REGISTRY = Path("/var/lib/ullm/served-model-campaign-claims")
FIXED_OUTCOME_REGISTRY = Path("/var/lib/ullm/served-model-campaign-outcomes")
FIXED_ACTIVE_MANIFEST = Path("/etc/ullm/served-models/active.json")
FIXED_SYSTEMD_UNIT_PATH = Path(
    "/etc/systemd/system/ullm-openai.service"
)
FIXED_ENVIRONMENT_FILE_PATH = Path(
    "/etc/ullm/openai-gateway-manifest.env"
)
FIXED_SERVICE_UNIT = "ullm-openai.service"
FIXED_OPENWEBUI_IMAGE = (
    "ullm/open-webui@sha256:"
    "ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
FIXED_BROWSER_IMAGE = (
    "sha256:"
    "0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea"
)
FIXED_OPENWEBUI_CONFIG_IMAGE = "ullm/open-webui:0.9.4-ullm.1"
FIXED_OPENWEBUI_CONTAINER_NAME = "open-webui"
RENAME_NOREPLACE = 1
MAX_DOCUMENT_BYTES = 1_048_576
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

AUTHORIZATION_FIELDS = {
    "schema_version",
    "authorization_id",
    "issued_at",
    "expires_at",
    "max_attempts",
    "authorization_note",
    "purpose",
    "required_final_route",
    "source",
    "before",
    "aq4_release",
    "candidate",
    "campaigns",
    "rollback",
    "prior_outcome",
}
SOURCE_FIELDS = {"commit", "tree"}
BEFORE_FIELDS = {
    "model_id",
    "format_id",
    "manifest_sha256",
    "worker_protocol",
    "worker_binary_path",
    "worker_binary_sha256",
    "promotion_source_commit",
    "promotion_receipt_path",
    "promotion_receipt_sha256",
}
CANDIDATE_FIELDS = {
    "model_id",
    "format_id",
    "manifest_sha256",
    "worker_protocol",
    "worker_binary_sha256",
    "promotion_source_commit",
    "promotion_receipt_sha256",
}
AQ4_RELEASE_FIELDS = {
    "source",
    "openwebui_image",
    "promotion_evidence",
    "promotion_receipt",
    "release_evidence_path",
    "release_validator_path",
    "browser_validator_path",
}
AQ4_SOURCE_FIELDS = {"root", "commit", "tree"}
ARTIFACT_REFERENCE_FIELDS = {"source_path", "path", "sha256"}
CAMPAIGN_FIELDS = {
    "aq4_reasoning_release",
    "aq4_reasoning_browser",
    "aq4_bundle",
    "sq8_full",
    "reasoning_release",
    "reasoning_browser",
}
CAMPAIGN_IDENTITY_FIELDS = {"run_id", "final_path"}
ROLLBACK_FIELDS = {
    "backup_path",
    "systemd_unit_sha256",
    "environment_sha256",
}
PRIOR_OUTCOME_FIELDS = {"path", "sha256"}
CLAIM_FIELDS = {
    "schema_version",
    "authorization_id",
    "authorization_path",
    "authorization_sha256",
    "claimed_at",
    "attempt",
    "max_attempts",
}
OUTCOME_FIELDS = {
    "schema_version",
    "authorization_id",
    "authorization_path",
    "authorization_sha256",
    "claim_path",
    "claim_sha256",
    "started_at",
    "completed_at",
    "status",
    "failure_stage",
    "stages",
    "aq4_observations",
    "candidate_observations",
    "campaigns",
    "restoration",
}
OUTCOME_STAGE_FIELDS = {
    "claim",
    "lock",
    "preflight",
    "backup",
    "candidate_activation",
    "candidate_reconciliation",
    "candidate_checks",
    "sq8_full",
    "reasoning_release",
    "reasoning_browser",
    "aq4_restore",
    "reverse_reconciliation",
    "aq4_reasoning_release",
    "aq4_reasoning_browser",
    "aq4_bundle",
    "final_checks",
}
OUTCOME_STAGE_STATES = {"pending", "passed", "failed", "skipped"}
OUTCOME_STATUSES = {
    "succeeded_restored",
    "failed_restored",
    "failed_restore",
}
OUTCOME_OBSERVATION_FIELDS = {
    "stage",
    "active_manifest_sha256",
    "bytes_equal",
}
OUTCOME_CAMPAIGN_FIELDS = {
    "run_id",
    "path",
    "kind",
    "sha256",
    "artifact_count",
    "total_bytes",
    "selected_artifacts",
}
OUTCOME_RESTORATION_FIELDS = {
    "expected_manifest_sha256",
    "displaced_manifest_sha256",
    "observed_manifest_sha256",
    "bytes_equal",
    "reverse_reconciliation_passed",
    "final_checks_passed",
    "model_id",
    "format_id",
    "worker_binary_sha256",
    "proof",
}
RECOVERY_FIELDS = {
    "schema_version",
    "authorization_id",
    "authorization_path",
    "authorization_sha256",
    "claim_path",
    "claim_sha256",
    "started_at",
    "completed_at",
    "status",
    "failure_stage",
    "source",
    "active_before",
    "backup",
    "restoration",
}
RECOVERY_ACTIVE_FIELDS = {"path", "sha256", "state"}
RECOVERY_BACKUP_FIELDS = {"path", "sha256"}
RECOVERY_STAGES = {
    "preflight",
    "aq4_restore",
    "reverse_reconciliation",
    "final_checks",
}
CANDIDATE_OBSERVATION_STAGES = (
    "candidate_activation",
    "candidate_reconciliation",
    "candidate_checks",
    "sq8_full:before",
    "sq8_full:after",
    "reasoning_release:before",
    "reasoning_release:after",
    "reasoning_browser:before",
    "reasoning_browser:after",
)
AQ4_OBSERVATION_STAGES = (
    "aq4_reasoning_release:before",
    "aq4_reasoning_release:after",
    "aq4_reasoning_browser:before",
    "aq4_reasoning_browser:after",
    "aq4_bundle:before",
    "aq4_bundle:after",
)
SAFE_ARTIFACT_NAME_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}\Z"
)


class AuthorizationError(ValueError):
    """Raised when an authorization or claim is unsafe or semantically invalid."""


class AuthorizationConsumed(AuthorizationError):
    """Raised when the authorization-derived claim already exists."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    mode: int
    uid: int
    nlink: int


@dataclass(frozen=True, slots=True)
class AuthorizationRecord:
    snapshot: FileSnapshot
    document: dict[str, Any]
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    snapshot: FileSnapshot
    document: dict[str, Any]
    authorization: AuthorizationRecord


@dataclass(frozen=True, slots=True)
class RegistryPolicy:
    claim_registry: Path = FIXED_CLAIM_REGISTRY
    outcome_registry: Path = FIXED_OUTCOME_REGISTRY
    required_uid: int = 0
    active_manifest_path: Path = FIXED_ACTIVE_MANIFEST
    systemd_unit_path: Path = FIXED_SYSTEMD_UNIT_PATH
    environment_file_path: Path = FIXED_ENVIRONMENT_FILE_PATH
    service_unit: str = FIXED_SERVICE_UNIT


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise AuthorizationError("JSON contains a non-finite number")


def canonical_json_bytes(document: dict[str, Any]) -> bytes:
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
        raise AuthorizationError("document is not canonicalizable JSON") from error


def strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthorizationError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise AuthorizationError(f"{label} root must be an object")
    return document


def _reject_symlink_components(
    path: Path, label: str, *, leaf_may_absent: bool
) -> None:
    _lexical_absolute(path, label)
    current = Path(path.anchor)
    components = path.parts[1:]
    for index, component in enumerate(components):
        if component in {"", ".", ".."}:
            raise AuthorizationError(f"{label} path is not canonical")
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(components) - 1:
                return
            raise AuthorizationError(f"{label} has an absent path component") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise AuthorizationError(f"{label} traverses a symlink")


def _lexical_absolute(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise AuthorizationError(f"{label} path must be absolute")
    raw = os.fspath(path)
    normalized = Path(os.path.abspath(path))
    if (
        path.anchor != "/"
        or raw.startswith("//")
        or normalized != path
        or path.name in {"", ".", ".."}
        or ".." in path.parts
    ):
        raise AuthorizationError(f"{label} path is not canonical")
    return normalized


def _path_value(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AuthorizationError(f"{label} is invalid")
    path = Path(value)
    if value != os.fspath(path):
        raise AuthorizationError(f"{label} path is not canonical")
    return _lexical_absolute(path, label)


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise AuthorizationError("safe directory descriptor flags are unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_anchor(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
    )


def _open_directory(path: Path, label: str) -> int:
    absolute = _lexical_absolute(path / "_entry", label).parent
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _directory_flags())
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except AuthorizationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise AuthorizationError(
            f"{label} parent is unavailable or traverses a symlink"
        ) from error


def _open_parent(path: Path, label: str) -> tuple[Path, int, tuple[int, ...]]:
    absolute = _lexical_absolute(path, label)
    descriptor = _open_directory(absolute.parent, label)
    return absolute, descriptor, _stat_identity(os.fstat(descriptor))


def _stable_read(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_DOCUMENT_BYTES,
    required_mode: int | None = None,
    required_uid: int | None = None,
    required_nlink: int | None = None,
) -> FileSnapshot:
    absolute, parent, parent_before = _open_parent(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    verification_parent = -1
    descriptor = -1
    try:
        entry_before = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        descriptor = os.open(absolute.name, flags, dir_fd=parent)
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            _stat_identity(entry_before) != _stat_identity(before)
            or
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
            or (required_mode is not None and mode != required_mode)
            or (required_uid is not None and before.st_uid != required_uid)
            or (required_nlink is not None and before.st_nlink != required_nlink)
        ):
            raise AuthorizationError(f"{label} metadata is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise AuthorizationError(f"{label} exceeds its size bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        entry_after = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        parent_after = _stat_identity(os.fstat(parent))
        verification_parent = _open_directory(absolute.parent, label)
        parent_by_path = _stat_identity(os.fstat(verification_parent))
        raw = b"".join(chunks)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(entry_after)
            or parent_before != parent_after
            or parent_before != parent_by_path
            or len(raw) != before.st_size
        ):
            raise AuthorizationError(f"{label} changed while being read")
        return FileSnapshot(
            path=absolute,
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            mode=mode,
            uid=before.st_uid,
            nlink=before.st_nlink,
        )
    except AuthorizationError:
        raise
    except OSError as error:
        raise AuthorizationError(f"{label} is unavailable or changed") from error
    finally:
        for value in (verification_parent, descriptor, parent):
            if value >= 0:
                os.close(value)


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AuthorizationError(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a lowercase SHA-256")
    return value


def _git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a full lowercase Git object ID")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} is invalid")
    return value


def _bounded_text(value: Any, label: str, maximum: int = 4_096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise AuthorizationError(f"{label} is invalid")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise AuthorizationError(f"{label} is invalid") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise AuthorizationError(f"{label} is not canonical")
    return parsed


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise AuthorizationError("timestamp must be timezone-aware")
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _absolute_future_path(value: Any, label: str) -> Path:
    path = _path_value(value, label)
    _reject_symlink_components(path, label, leaf_may_absent=True)
    if path.exists() or path.is_symlink():
        raise AuthorizationError(f"{label} must name a fresh output")
    return path


def _absolute_bound_path(value: Any, label: str, *, require_fresh: bool) -> Path:
    if require_fresh:
        return _absolute_future_path(value, label)
    path = _path_value(value, label)
    if path.exists() or path.is_symlink():
        _reject_symlink_components(path, label, leaf_may_absent=False)
    else:
        _reject_symlink_components(path, label, leaf_may_absent=True)
    return path


def _absolute_archival_path(value: Any, label: str) -> Path:
    return _path_value(value, label)


def _nullable_identifier(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, label)


def _content_addressed_image(value: Any, label: str) -> str:
    text = _bounded_text(value, label, 1_024)
    marker = "@sha256:"
    if marker not in text or HASH_RE.fullmatch(text.rsplit(marker, 1)[1]) is None:
        raise AuthorizationError(f"{label} is not content-addressed")
    return text


def _bound_existing_path(
    value: Any,
    label: str,
    *,
    directory: bool,
) -> Path:
    path = _absolute_bound_path(value, label, require_fresh=False)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AuthorizationError(f"{label} is unavailable") from error
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(metadata.st_mode) or not expected(metadata.st_mode):
        raise AuthorizationError(f"{label} has the wrong file type")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_outcome_document_shape(document: dict[str, Any]) -> None:
    _exact_object(document, OUTCOME_FIELDS, "campaign outcome")
    if document["schema_version"] != OUTCOME_SCHEMA:
        raise AuthorizationError("campaign outcome schema differs")
    _identifier(document["authorization_id"], "outcome.authorization_id")
    if not isinstance(document["authorization_path"], str):
        raise AuthorizationError("outcome.authorization_path is invalid")
    _absolute_bound_path(
        document["authorization_path"],
        "outcome.authorization_path",
        require_fresh=False,
    )
    _hash(document["authorization_sha256"], "outcome.authorization_sha256")
    if not isinstance(document["claim_path"], str):
        raise AuthorizationError("outcome.claim_path is invalid")
    _absolute_bound_path(
        document["claim_path"],
        "outcome.claim_path",
        require_fresh=False,
    )
    _hash(document["claim_sha256"], "outcome.claim_sha256")
    started_at = _timestamp(document["started_at"], "outcome.started_at")
    completed_at = _timestamp(document["completed_at"], "outcome.completed_at")
    if completed_at < started_at:
        raise AuthorizationError("campaign outcome completion precedes its start")
    if document["status"] not in OUTCOME_STATUSES:
        raise AuthorizationError("campaign outcome status differs")
    failure_stage = document["failure_stage"]
    if failure_stage is not None and (
        not isinstance(failure_stage, str)
        or failure_stage not in OUTCOME_STAGE_FIELDS
    ):
        raise AuthorizationError("campaign outcome failure stage differs")

    stages = _exact_object(
        document["stages"], OUTCOME_STAGE_FIELDS, "outcome.stages"
    )
    if any(value not in OUTCOME_STAGE_STATES for value in stages.values()):
        raise AuthorizationError("campaign outcome stage state differs")
    if "pending" in stages.values():
        raise AuthorizationError("campaign outcome retains a pending stage")
    if stages["claim"] != "passed":
        raise AuthorizationError("campaign outcome lacks its consumed claim")
    if document["status"] == "succeeded_restored":
        if failure_stage is not None or any(value != "passed" for value in stages.values()):
            raise AuthorizationError("successful campaign outcome has incomplete stages")
    elif failure_stage is None or stages[failure_stage] != "failed":
        raise AuthorizationError("failed campaign outcome lacks its failed stage")

    for observation_field in ("aq4_observations", "candidate_observations"):
        observations = document[observation_field]
        if (
            not isinstance(observations, list)
            or len(observations) > 4_096
        ):
            raise AuthorizationError("campaign outcome observations are invalid")
        for index, value in enumerate(observations):
            observation = _exact_object(
                value,
                OUTCOME_OBSERVATION_FIELDS,
                f"outcome.{observation_field}[{index}]",
            )
            _identifier(
                observation["stage"],
                f"outcome.{observation_field}[{index}].stage",
            )
            _hash(
                observation["active_manifest_sha256"],
                f"outcome.{observation_field}[{index}].active_manifest_sha256",
            )
            if type(observation["bytes_equal"]) is not bool:
                raise AuthorizationError(
                    "campaign outcome observation result is invalid"
                )

    campaigns = _exact_object(
        document["campaigns"], CAMPAIGN_FIELDS, "outcome.campaigns"
    )
    for name in sorted(CAMPAIGN_FIELDS):
        value = campaigns[name]
        if value is None:
            continue
        campaign = _exact_object(
            value, OUTCOME_CAMPAIGN_FIELDS, f"outcome.campaigns.{name}"
        )
        _identifier(campaign["run_id"], f"outcome.campaigns.{name}.run_id")
        if not isinstance(campaign["path"], str):
            raise AuthorizationError(f"outcome.campaigns.{name}.path is invalid")
        _absolute_archival_path(
            campaign["path"],
            f"outcome.campaigns.{name}.path",
        )
        if campaign["kind"] not in {"file", "directory"}:
            raise AuthorizationError(f"outcome.campaigns.{name}.kind differs")
        _hash(campaign["sha256"], f"outcome.campaigns.{name}.sha256")
        for field in ("artifact_count", "total_bytes"):
            if (
                type(campaign[field]) is not int
                or campaign[field] < 1
                or campaign[field] > (1 << 63) - 1
            ):
                raise AuthorizationError(
                    f"outcome.campaigns.{name}.{field} is invalid"
                )
        selected = campaign["selected_artifacts"]
        if not isinstance(selected, dict) or len(selected) > 64:
            raise AuthorizationError(
                f"outcome.campaigns.{name}.selected_artifacts is invalid"
            )
        for artifact_name, digest in selected.items():
            if (
                not isinstance(artifact_name, str)
                or SAFE_ARTIFACT_NAME_RE.fullmatch(artifact_name) is None
                or artifact_name.startswith("/")
                or ".." in Path(artifact_name).parts
            ):
                raise AuthorizationError(
                    f"outcome.campaigns.{name} selected artifact name is invalid"
                )
            _hash(
                digest,
                f"outcome.campaigns.{name}.selected_artifacts.{artifact_name}",
            )
    if document["status"] == "succeeded_restored" and any(
        campaigns[name] is None for name in CAMPAIGN_FIELDS
    ):
        raise AuthorizationError("successful campaign outcome lacks campaign output")

    restoration = _exact_object(
        document["restoration"],
        OUTCOME_RESTORATION_FIELDS,
        "outcome.restoration",
    )
    expected = _hash(
        restoration["expected_manifest_sha256"],
        "outcome.restoration.expected_manifest_sha256",
    )
    observed = restoration["observed_manifest_sha256"]
    displaced = restoration["displaced_manifest_sha256"]
    if displaced is not None:
        _hash(displaced, "outcome.restoration.displaced_manifest_sha256")
    if observed is not None:
        _hash(observed, "outcome.restoration.observed_manifest_sha256")
    for field in (
        "bytes_equal",
        "reverse_reconciliation_passed",
        "final_checks_passed",
    ):
        if type(restoration[field]) is not bool:
            raise AuthorizationError(f"outcome.restoration.{field} is invalid")
    model_id = _nullable_identifier(
        restoration["model_id"], "outcome.restoration.model_id"
    )
    format_id = _nullable_identifier(
        restoration["format_id"], "outcome.restoration.format_id"
    )
    worker_hash = restoration["worker_binary_sha256"]
    if worker_hash is not None:
        _hash(worker_hash, "outcome.restoration.worker_binary_sha256")
    if restoration["bytes_equal"] != (observed == expected):
        raise AuthorizationError("campaign outcome restoration byte result differs")
    proof = restoration["proof"]
    if proof is not None and not isinstance(proof, dict):
        raise AuthorizationError("campaign outcome restoration proof is invalid")
    if document["status"] in {"succeeded_restored", "failed_restored"}:
        if (
            not restoration["bytes_equal"]
            or not restoration["reverse_reconciliation_passed"]
            or not restoration["final_checks_passed"]
            or model_id != "ullm-qwen3.5-9b-aq4"
            or format_id != "AQ4_0"
            or worker_hash is None
            or proof is None
        ):
            raise AuthorizationError("campaign outcome does not prove AQ4 restoration")
    elif (
        restoration["bytes_equal"]
        and restoration["reverse_reconciliation_passed"]
        and restoration["final_checks_passed"]
    ):
        raise AuthorizationError("failed-restore outcome reports complete restoration")


def validate_outcome_document(
    document: dict[str, Any],
    *,
    claim: ClaimRecord | None = None,
    policy: RegistryPolicy = RegistryPolicy(),
) -> None:
    """Validate one outcome and, when supplied, bind it to the consumed claim."""

    _validate_outcome_document_shape(document)
    if claim is None:
        return
    authorization = claim.authorization
    if (
        document["authorization_id"]
        != authorization.document["authorization_id"]
        or document["authorization_path"]
        != os.fspath(authorization.snapshot.path)
        or document["authorization_sha256"] != authorization.snapshot.sha256
        or document["claim_path"] != os.fspath(claim.snapshot.path)
        or document["claim_sha256"] != claim.snapshot.sha256
        or document["restoration"]["expected_manifest_sha256"]
        != authorization.document["before"]["manifest_sha256"]
    ):
        raise AuthorizationError("campaign outcome claim identity differs")
    if _timestamp(document["started_at"], "outcome.started_at") < _timestamp(
        claim.document["claimed_at"], "claim.claimed_at"
    ):
        raise AuthorizationError("campaign outcome predates its claim")
    observations = document["candidate_observations"]
    observed_stages = tuple(value["stage"] for value in observations)
    if (
        observed_stages
        != CANDIDATE_OBSERVATION_STAGES[: len(observed_stages)]
        or any(
            value["bytes_equal"] is not True
            or value["active_manifest_sha256"]
            != authorization.document["candidate"]["manifest_sha256"]
            for value in observations
        )
    ):
        raise AuthorizationError(
            "campaign outcome candidate observations differ from authorization"
        )
    if (
        document["status"] == "succeeded_restored"
        and observed_stages != CANDIDATE_OBSERVATION_STAGES
    ):
        raise AuthorizationError(
            "successful campaign outcome lacks complete candidate observations"
        )
    aq4_observations = document["aq4_observations"]
    aq4_observed_stages = tuple(
        value["stage"] for value in aq4_observations
    )
    if (
        aq4_observed_stages
        != AQ4_OBSERVATION_STAGES[: len(aq4_observed_stages)]
        or any(
            value["bytes_equal"] is not True
            or value["active_manifest_sha256"]
            != authorization.document["before"]["manifest_sha256"]
            for value in aq4_observations
        )
    ):
        raise AuthorizationError(
            "campaign outcome AQ4 observations differ from authorization"
        )
    if (
        document["status"] == "succeeded_restored"
        and aq4_observed_stages != AQ4_OBSERVATION_STAGES
    ):
        raise AuthorizationError(
            "successful campaign outcome lacks complete AQ4 observations"
        )
    for name in sorted(CAMPAIGN_FIELDS):
        campaign = document["campaigns"][name]
        stage_passed = document["stages"][name] == "passed"
        if stage_passed != (campaign is not None):
            raise AuthorizationError(
                "campaign outcome stage/output presence differs"
            )
        if campaign is None:
            continue
        authorized = authorization.document["campaigns"][name]
        if (
            campaign["run_id"] != authorized["run_id"]
            or campaign["path"] != authorized["final_path"]
        ):
            raise AuthorizationError("campaign outcome run/output identity differs")
    if (
        document["status"] in {"succeeded_restored", "failed_restored"}
        and document["restoration"]["worker_binary_sha256"]
        != authorization.document["before"]["worker_binary_sha256"]
    ):
        raise AuthorizationError(
            "campaign outcome restored worker differs from authorization"
        )
    proof = document["restoration"]["proof"]
    if proof is not None:
        try:
            proof_active = proof.get("active_manifest")
            proof_service = proof.get("service")
            if not isinstance(proof_active, dict) or not isinstance(proof_service, dict):
                raise restoration_proof.RestorationProofError(
                    "restoration proof identity is missing"
                )
            restoration_proof.validate_proof(
                proof,
                authorization_sha256=authorization.snapshot.sha256,
                claim_sha256=claim.snapshot.sha256,
                active_manifest_path=policy.active_manifest_path,
                expected_manifest_sha256=authorization.document["before"][
                    "manifest_sha256"
                ],
                expected_worker_sha256=authorization.document["before"][
                    "worker_binary_sha256"
                ],
                service_unit=policy.service_unit,
            )
        except restoration_proof.RestorationProofError as error:
            raise AuthorizationError(
                "campaign outcome live restoration proof differs"
            ) from error


def validate_recovery_document(
    document: dict[str, Any],
    *,
    claim: ClaimRecord,
    policy: RegistryPolicy = RegistryPolicy(),
) -> None:
    """Strictly bind one immutable recovery receipt to a consumed claim."""

    _exact_object(document, RECOVERY_FIELDS, "campaign recovery receipt")
    if (
        document["schema_version"] != RECOVERY_SCHEMA
        or document["authorization_id"]
        != claim.authorization.document["authorization_id"]
        or document["authorization_path"]
        != os.fspath(claim.authorization.snapshot.path)
        or document["authorization_sha256"]
        != claim.authorization.snapshot.sha256
        or document["claim_path"] != os.fspath(claim.snapshot.path)
        or document["claim_sha256"] != claim.snapshot.sha256
    ):
        raise AuthorizationError("campaign recovery claim identity differs")
    started_at = _timestamp(document["started_at"], "recovery.started_at")
    completed_at = _timestamp(document["completed_at"], "recovery.completed_at")
    claimed_at = _timestamp(claim.document["claimed_at"], "claim.claimed_at")
    if completed_at < started_at or started_at < claimed_at:
        raise AuthorizationError("campaign recovery timestamps differ")
    if document["status"] not in {"restored", "failed_restore"}:
        raise AuthorizationError("campaign recovery status differs")
    failure_stage = document["failure_stage"]
    if (
        failure_stage is not None
        and (
            not isinstance(failure_stage, str)
            or failure_stage not in RECOVERY_STAGES
        )
    ):
        raise AuthorizationError("campaign recovery failure stage differs")
    if (document["status"] == "restored") != (failure_stage is None):
        raise AuthorizationError("campaign recovery failure identity differs")

    source = _exact_object(document["source"], SOURCE_FIELDS, "recovery.source")
    if source != claim.authorization.document["source"]:
        raise AuthorizationError("campaign recovery source identity differs")

    active = _exact_object(
        document["active_before"],
        RECOVERY_ACTIVE_FIELDS,
        "recovery.active_before",
    )
    if not isinstance(active["path"], str):
        raise AuthorizationError("campaign recovery active path is invalid")
    _absolute_bound_path(
        active["path"],
        "recovery.active_before.path",
        require_fresh=False,
    )
    if active["path"] != os.fspath(policy.active_manifest_path):
        raise AuthorizationError(
            "campaign recovery active path differs from policy"
        )
    active_hash = _hash(active["sha256"], "recovery.active_before.sha256")
    if active["state"] == "aq4":
        expected_active_hash = claim.authorization.document["before"][
            "manifest_sha256"
        ]
    elif active["state"] == "sq8":
        expected_active_hash = claim.authorization.document["candidate"][
            "manifest_sha256"
        ]
    elif active["state"] == "unknown":
        expected_active_hash = active_hash
    else:
        raise AuthorizationError("campaign recovery active state differs")
    if active_hash != expected_active_hash:
        raise AuthorizationError("campaign recovery active identity differs")

    backup = _exact_object(
        document["backup"], RECOVERY_BACKUP_FIELDS, "recovery.backup"
    )
    if (
        backup["path"]
        != claim.authorization.document["rollback"]["backup_path"]
        or _hash(backup["sha256"], "recovery.backup.sha256")
        != claim.authorization.document["before"]["manifest_sha256"]
    ):
        raise AuthorizationError("campaign recovery backup identity differs")

    restoration = _exact_object(
        document["restoration"],
        OUTCOME_RESTORATION_FIELDS,
        "recovery.restoration",
    )
    expected = _hash(
        restoration["expected_manifest_sha256"],
        "recovery.restoration.expected_manifest_sha256",
    )
    observed = restoration["observed_manifest_sha256"]
    displaced = restoration["displaced_manifest_sha256"]
    if displaced is not None:
        _hash(displaced, "recovery.restoration.displaced_manifest_sha256")
    if observed is not None:
        _hash(observed, "recovery.restoration.observed_manifest_sha256")
    for field in (
        "bytes_equal",
        "reverse_reconciliation_passed",
        "final_checks_passed",
    ):
        if type(restoration[field]) is not bool:
            raise AuthorizationError(
                f"recovery.restoration.{field} is invalid"
            )
    worker_hash = restoration["worker_binary_sha256"]
    if worker_hash is not None:
        _hash(worker_hash, "recovery.restoration.worker_binary_sha256")
    if (
        expected != claim.authorization.document["before"]["manifest_sha256"]
        or restoration["bytes_equal"] != (observed == expected)
        or (
            restoration["model_id"] is not None
            and restoration["model_id"] != "ullm-qwen3.5-9b-aq4"
        )
        or (
            restoration["format_id"] is not None
            and restoration["format_id"] != "AQ4_0"
        )
        or (
            worker_hash is not None
            and worker_hash
            != claim.authorization.document["before"]["worker_binary_sha256"]
        )
        or (
            restoration["proof"] is not None
            and not isinstance(restoration["proof"], dict)
        )
    ):
        raise AuthorizationError("campaign recovery restoration identity differs")

    fully_restored = (
        restoration["bytes_equal"]
        and restoration["reverse_reconciliation_passed"]
        and restoration["final_checks_passed"]
        and restoration["model_id"] == "ullm-qwen3.5-9b-aq4"
        and restoration["format_id"] == "AQ4_0"
        and worker_hash
        == claim.authorization.document["before"]["worker_binary_sha256"]
        and isinstance(restoration["proof"], dict)
    )
    if (document["status"] == "restored") != fully_restored:
        raise AuthorizationError("campaign recovery restoration result differs")
    if restoration["proof"] is not None:
        proof = restoration["proof"]
        try:
            proof_active = proof.get("active_manifest")
            proof_service = proof.get("service")
            if not isinstance(proof_active, dict) or not isinstance(
                proof_service, dict
            ):
                raise restoration_proof.RestorationProofError(
                    "restoration proof identity is missing"
                )
            restoration_proof.validate_proof(
                proof,
                authorization_sha256=claim.authorization.snapshot.sha256,
                claim_sha256=claim.snapshot.sha256,
                active_manifest_path=policy.active_manifest_path,
                expected_manifest_sha256=claim.authorization.document["before"][
                    "manifest_sha256"
                ],
                expected_worker_sha256=claim.authorization.document["before"][
                    "worker_binary_sha256"
                ],
                service_unit=policy.service_unit,
            )
        except restoration_proof.RestorationProofError as error:
            raise AuthorizationError(
                "campaign recovery live restoration proof differs"
            ) from error


def _load_claim_for_record(
    authorization: AuthorizationRecord,
    *,
    policy: RegistryPolicy,
) -> ClaimRecord:
    expected = claim_path(authorization.snapshot.sha256, policy=policy)
    snapshot = _stable_read(
        expected,
        "campaign authorization claim",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign authorization claim")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError("campaign authorization claim is not canonical JSON")
    _exact_object(document, CLAIM_FIELDS, "campaign authorization claim")
    if (
        document["schema_version"] != CLAIM_SCHEMA
        or document["authorization_id"]
        != authorization.document["authorization_id"]
        or document["authorization_path"]
        != os.fspath(authorization.snapshot.path)
        or document["authorization_sha256"] != authorization.snapshot.sha256
        or document["attempt"] != 1
        or document["max_attempts"] != 1
    ):
        raise AuthorizationError("campaign authorization claim identity differs")
    claimed_at = _timestamp(document["claimed_at"], "claim.claimed_at")
    if claimed_at < authorization.issued_at or claimed_at >= authorization.expires_at:
        raise AuthorizationError("campaign authorization claim time is out of range")
    return ClaimRecord(snapshot, document, authorization)


def _validate_outcome_reference(
    value: Any,
    label: str,
    *,
    policy: RegistryPolicy,
) -> tuple[AuthorizationRecord, dict[str, Any]]:
    reference = _exact_object(value, PRIOR_OUTCOME_FIELDS, label)
    expected_hash = _hash(reference["sha256"], f"{label}.sha256")
    if not isinstance(reference["path"], str):
        raise AuthorizationError(f"{label}.path is invalid")
    snapshot = _stable_read(
        Path(reference["path"]),
        label,
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    if snapshot.sha256 != expected_hash:
        raise AuthorizationError(f"{label} SHA-256 differs")
    outcome = strict_json_bytes(snapshot.raw, label)
    if canonical_json_bytes(outcome) != snapshot.raw:
        raise AuthorizationError(f"{label} is not canonical JSON")
    _validate_outcome_document_shape(outcome)
    if outcome.get("status") not in {"failed_restored", "failed_restore"}:
        raise AuthorizationError(f"{label} is not a failed campaign outcome")
    authorization_snapshot = _stable_read(
        Path(outcome.get("authorization_path", "")),
        f"{label} authorization",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    if authorization_snapshot.sha256 != outcome.get("authorization_sha256"):
        raise AuthorizationError(f"{label} authorization SHA-256 differs")
    authorization_document = strict_json_bytes(
        authorization_snapshot.raw, f"{label} authorization"
    )
    if canonical_json_bytes(authorization_document) != authorization_snapshot.raw:
        raise AuthorizationError(f"{label} authorization is not canonical JSON")
    issued_at, expires_at = validate_authorization_document(
        authorization_document,
        now=datetime.fromtimestamp(0, timezone.utc),
        required_uid=policy.required_uid,
        validate_prior_outcome=False,
        require_fresh_outputs=False,
        require_bound_inputs=False,
        enforce_current_window=False,
        policy=policy,
    )
    authorization = AuthorizationRecord(
        authorization_snapshot,
        authorization_document,
        issued_at,
        expires_at,
    )
    claim = _load_claim_for_record(authorization, policy=policy)
    expected_outcome = outcome_path(authorization_snapshot.sha256, policy=policy)
    try:
        resolved_expected_outcome = expected_outcome.resolve(strict=True)
    except OSError as error:
        raise AuthorizationError(
            f"{label} fixed outcome path is unavailable"
        ) from error
    if snapshot.path != resolved_expected_outcome:
        raise AuthorizationError(f"{label} is outside the fixed outcome registry")
    try:
        validate_outcome_document(outcome, claim=claim, policy=policy)
    except AuthorizationError as error:
        raise AuthorizationError(f"{label} is invalid") from error
    return authorization, outcome


def validate_authorization_document(
    document: dict[str, Any],
    *,
    now: datetime,
    required_uid: int = 0,
    validate_prior_outcome: bool = True,
    require_fresh_outputs: bool = True,
    require_bound_inputs: bool = True,
    enforce_current_window: bool = True,
    policy: RegistryPolicy | None = None,
    source_root: Path | None = None,
) -> tuple[datetime, datetime]:
    _exact_object(document, AUTHORIZATION_FIELDS, "authorization")
    if document["schema_version"] != AUTHORIZATION_SCHEMA:
        raise AuthorizationError("authorization schema differs")
    _identifier(document["authorization_id"], "authorization_id")
    issued_at = _timestamp(document["issued_at"], "issued_at")
    expires_at = _timestamp(document["expires_at"], "expires_at")
    if expires_at <= issued_at:
        raise AuthorizationError("authorization window is invalid")
    if enforce_current_window:
        normalized_now = now.astimezone(timezone.utc)
        if issued_at > normalized_now:
            raise AuthorizationError("authorization is not yet valid")
        if expires_at <= normalized_now:
            raise AuthorizationError("authorization is expired")
    if type(document["max_attempts"]) is not int or document["max_attempts"] != 1:
        raise AuthorizationError("authorization max_attempts must equal one")
    _bounded_text(document["authorization_note"], "authorization_note")
    if document["purpose"] != "temporary_candidate_active_evidence_collection_only":
        raise AuthorizationError("authorization purpose differs")
    if (
        document["required_final_route"]
        != "restore_exact_aq4_then_bundle_v2_activation"
    ):
        raise AuthorizationError("authorization final route differs")

    source = _exact_object(document["source"], SOURCE_FIELDS, "source")
    _git_object(source["commit"], "source.commit")
    _git_object(source["tree"], "source.tree")

    before = _exact_object(document["before"], BEFORE_FIELDS, "before")
    if (
        before["model_id"] != "ullm-qwen3.5-9b-aq4"
        or before["format_id"] != "AQ4_0"
        or before["worker_protocol"] != "ullm.worker.v2"
    ):
        raise AuthorizationError("authorization before identity is not AQ4_0")
    _hash(before["manifest_sha256"], "before.manifest_sha256")
    existing_or_bound = (
        _bound_existing_path
        if require_bound_inputs
        else lambda value, label, *, directory: _absolute_archival_path(
            value,
            label,
        )
    )
    before_worker = existing_or_bound(
        before["worker_binary_path"],
        "before.worker_binary_path",
        directory=False,
    )
    _hash(before["worker_binary_sha256"], "before.worker_binary_sha256")
    _git_object(before["promotion_source_commit"], "before.promotion_source_commit")
    before_receipt = existing_or_bound(
        before["promotion_receipt_path"],
        "before.promotion_receipt_path",
        directory=False,
    )
    _hash(
        before["promotion_receipt_sha256"],
        "before.promotion_receipt_sha256",
    )

    aq4_release = _exact_object(
        document["aq4_release"],
        AQ4_RELEASE_FIELDS,
        "aq4_release",
    )
    aq4_source = _exact_object(
        aq4_release["source"],
        AQ4_SOURCE_FIELDS,
        "aq4_release.source",
    )
    aq4_source_root = existing_or_bound(
        aq4_source["root"],
        "aq4_release.source.root",
        directory=True,
    )
    aq4_source_commit = _git_object(
        aq4_source["commit"],
        "aq4_release.source.commit",
    )
    _git_object(aq4_source["tree"], "aq4_release.source.tree")
    if aq4_source_commit != before["promotion_source_commit"]:
        raise AuthorizationError(
            "AQ4 release source/promotion commit differs"
        )
    openwebui_image = _content_addressed_image(
        aq4_release["openwebui_image"],
        "aq4_release.openwebui_image",
    )
    if openwebui_image != FIXED_OPENWEBUI_IMAGE:
        raise AuthorizationError("AQ4 release OpenWebUI image differs")
    aq4_promotion_source_paths: dict[str, Path] = {}
    aq4_promotion_paths: dict[str, Path] = {}
    for name in ("promotion_evidence", "promotion_receipt"):
        reference = _exact_object(
            aq4_release[name],
            ARTIFACT_REFERENCE_FIELDS,
            f"aq4_release.{name}",
        )
        aq4_promotion_source_paths[name] = existing_or_bound(
            reference["source_path"],
            f"aq4_release.{name}.source_path",
            directory=False,
        )
        aq4_promotion_paths[name] = (
            _absolute_archival_path(
                reference["path"],
                f"aq4_release.{name}.path",
            )
            if not require_bound_inputs and not require_fresh_outputs
            else _absolute_bound_path(
                reference["path"],
                f"aq4_release.{name}.path",
                require_fresh=require_fresh_outputs,
            )
        )
        _hash(reference["sha256"], f"aq4_release.{name}.sha256")
    if (
        aq4_promotion_source_paths["promotion_receipt"] != before_receipt
        or aq4_release["promotion_receipt"]["sha256"]
        != before["promotion_receipt_sha256"]
    ):
        raise AuthorizationError(
            "AQ4 release/before promotion receipt differs"
        )
    aq4_fresh_paths: dict[str, Path] = {}
    for name in (
        "release_evidence_path",
        "release_validator_path",
        "browser_validator_path",
    ):
        aq4_fresh_paths[name] = (
            _absolute_archival_path(
                aq4_release[name],
                f"aq4_release.{name}",
            )
            if not require_bound_inputs and not require_fresh_outputs
            else _absolute_bound_path(
                aq4_release[name],
                f"aq4_release.{name}",
                require_fresh=require_fresh_outputs,
            )
        )

    candidate = _exact_object(
        document["candidate"], CANDIDATE_FIELDS, "candidate"
    )
    if (
        candidate["model_id"] != "ullm-qwen3-14b-sq8"
        or candidate["format_id"] != "SQ8_0"
        or candidate["worker_protocol"] != "ullm.worker.v2"
    ):
        raise AuthorizationError("authorization candidate identity is not SQ8_0 v2")
    _hash(candidate["manifest_sha256"], "candidate.manifest_sha256")
    _hash(
        candidate["worker_binary_sha256"], "candidate.worker_binary_sha256"
    )
    _git_object(
        candidate["promotion_source_commit"],
        "candidate.promotion_source_commit",
    )
    _hash(
        candidate["promotion_receipt_sha256"],
        "candidate.promotion_receipt_sha256",
    )
    if source["commit"] != candidate["promotion_source_commit"]:
        raise AuthorizationError("authorization source/candidate commit differs")
    if before["manifest_sha256"] == candidate["manifest_sha256"]:
        raise AuthorizationError("authorization before and candidate manifests are equal")

    campaigns = _exact_object(document["campaigns"], CAMPAIGN_FIELDS, "campaigns")
    final_paths: set[Path] = set()
    run_ids: set[str] = set()
    for name in sorted(CAMPAIGN_FIELDS):
        campaign = _exact_object(
            campaigns[name], CAMPAIGN_IDENTITY_FIELDS, f"campaigns.{name}"
        )
        run_id = _identifier(campaign["run_id"], f"campaigns.{name}.run_id")
        final_path = (
            _absolute_archival_path(
                campaign["final_path"],
                f"campaigns.{name}.final_path",
            )
            if not require_bound_inputs and not require_fresh_outputs
            else _absolute_bound_path(
                campaign["final_path"],
                f"campaigns.{name}.final_path",
                require_fresh=require_fresh_outputs,
            )
        )
        if run_id in run_ids or final_path in final_paths:
            raise AuthorizationError("campaign run IDs and final paths must be distinct")
        run_ids.add(run_id)
        final_paths.add(final_path)

    aq4_bundle_root = Path(
        campaigns["aq4_bundle"]["final_path"]
    ).parent
    aq4_component_paths = {
        aq4_promotion_paths["promotion_evidence"],
        aq4_promotion_paths["promotion_receipt"],
        aq4_fresh_paths["release_evidence_path"],
        aq4_fresh_paths["release_validator_path"],
        Path(campaigns["aq4_reasoning_browser"]["final_path"]),
        aq4_fresh_paths["browser_validator_path"],
    }
    if any(
        path != aq4_bundle_root and aq4_bundle_root not in path.parents
        for path in aq4_component_paths
    ):
        raise AuthorizationError(
            "AQ4 bundle components must be below its output parent"
        )
    if (
        len(aq4_component_paths) != 6
        or len(set(aq4_promotion_source_paths.values())) != 2
        or any(
            path in aq4_promotion_source_paths.values()
            for path in aq4_promotion_paths.values()
        )
        or any(path in final_paths for path in aq4_fresh_paths.values())
    ):
        raise AuthorizationError("AQ4 release paths must be distinct")

    rollback = _exact_object(document["rollback"], ROLLBACK_FIELDS, "rollback")
    backup_path = (
        _absolute_archival_path(
            rollback["backup_path"],
            "rollback.backup_path",
        )
        if not require_bound_inputs and not require_fresh_outputs
        else _absolute_bound_path(
            rollback["backup_path"],
            "rollback.backup_path",
            require_fresh=require_fresh_outputs,
        )
    )
    if backup_path in final_paths:
        raise AuthorizationError("rollback backup collides with a campaign output")
    if source_root is not None:
        try:
            source_root_absolute = source_root.resolve(strict=True)
        except OSError as error:
            raise AuthorizationError(
                "authorization source root is unavailable"
            ) from error
        if _paths_overlap(source_root_absolute, aq4_source_root):
            raise AuthorizationError(
                "SQ8 and AQ4 source roots must be disjoint"
            )
        all_outputs = (
            *final_paths,
            *aq4_fresh_paths.values(),
            *aq4_promotion_paths.values(),
            backup_path,
        )
        for output_path in all_outputs:
            if (
                _paths_overlap(output_path, source_root_absolute)
                or _paths_overlap(output_path, aq4_source_root)
            ):
                raise AuthorizationError(
                    "campaign outputs must be outside the source root"
                )
        if any(
            _paths_overlap(left, right)
            for index, left in enumerate(all_outputs)
            for right in all_outputs[index + 1 :]
        ):
            raise AuthorizationError(
                "campaign output paths must not overlap"
            )
    if any(
        _paths_overlap(before_worker, path)
        for path in (
            *final_paths,
            *aq4_fresh_paths.values(),
            *aq4_promotion_paths.values(),
            backup_path,
        )
    ):
        raise AuthorizationError("AQ4 worker path collides with an output")
    _hash(rollback["systemd_unit_sha256"], "rollback.systemd_unit_sha256")
    _hash(rollback["environment_sha256"], "rollback.environment_sha256")

    prior_outcome = document["prior_outcome"]
    if prior_outcome is not None:
        if validate_prior_outcome:
            selected_policy = (
                RegistryPolicy(required_uid=required_uid)
                if policy is None
                else policy
            )
            previous_authorization, _previous_outcome = _validate_outcome_reference(
                prior_outcome,
                "prior_outcome",
                policy=selected_policy,
            )
            previous = previous_authorization.document
            if previous["source"] != document["source"]:
                raise AuthorizationError("prior_outcome source lineage differs")
            for field in (
                "model_id",
                "format_id",
                "manifest_sha256",
                "worker_protocol",
                "worker_binary_sha256",
                "promotion_source_commit",
                "promotion_receipt_sha256",
            ):
                if previous["before"][field] != document["before"][field]:
                    raise AuthorizationError(
                        "prior_outcome before lineage differs"
                    )
            if previous["candidate"] != document["candidate"]:
                raise AuthorizationError(
                    "prior_outcome candidate lineage differs"
                )
            previous_aq4 = previous["aq4_release"]
            selected_aq4 = document["aq4_release"]
            if (
                previous_aq4["source"]["commit"]
                != selected_aq4["source"]["commit"]
                or previous_aq4["source"]["tree"]
                != selected_aq4["source"]["tree"]
                or previous_aq4["openwebui_image"]
                != selected_aq4["openwebui_image"]
                or previous_aq4["promotion_evidence"]["sha256"]
                != selected_aq4["promotion_evidence"]["sha256"]
                or previous_aq4["promotion_receipt"]["sha256"]
                != selected_aq4["promotion_receipt"]["sha256"]
            ):
                raise AuthorizationError(
                    "prior_outcome AQ4 release lineage differs"
                )
        else:
            reference = _exact_object(
                prior_outcome, PRIOR_OUTCOME_FIELDS, "prior_outcome"
            )
            if not isinstance(reference["path"], str):
                raise AuthorizationError("prior_outcome.path is invalid")
            _hash(reference["sha256"], "prior_outcome.sha256")
    return issued_at, expires_at


def load_authorization(
    path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
    require_fresh_outputs: bool = True,
    enforce_current_window: bool = True,
    require_bound_inputs: bool = True,
    source_root: Path | None = None,
) -> AuthorizationRecord:
    snapshot = _stable_read(
        path,
        "campaign authorization",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign authorization")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError("campaign authorization is not canonical JSON")
    issued_at, expires_at = validate_authorization_document(
        document,
        now=now,
        required_uid=policy.required_uid,
        require_fresh_outputs=require_fresh_outputs,
        enforce_current_window=enforce_current_window,
        require_bound_inputs=require_bound_inputs,
        policy=policy,
        source_root=source_root,
    )
    return AuthorizationRecord(snapshot, document, issued_at, expires_at)


def _validate_registry(path: Path, label: str, *, required_uid: int) -> Path:
    descriptor = -1
    try:
        descriptor = _open_directory(path, label)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise AuthorizationError(f"{label} is unavailable") from error
    try:
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise AuthorizationError(f"{label} metadata is unsafe")
        return _lexical_absolute(path / "_entry", label).parent
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def claim_path(
    authorization_sha256: str,
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> Path:
    _hash(authorization_sha256, "authorization_sha256")
    return policy.claim_registry / f"{authorization_sha256}.claim.json"


def outcome_path(
    authorization_sha256: str,
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> Path:
    _hash(authorization_sha256, "authorization_sha256")
    return policy.outcome_registry / f"{authorization_sha256}.outcome.json"


def recovery_path(
    authorization_sha256: str,
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> Path:
    _hash(authorization_sha256, "authorization_sha256")
    return policy.outcome_registry / f"{authorization_sha256}.recovery.json"


def _publish_no_replace(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    required_uid: int,
    label: str,
) -> FileSnapshot:
    if os.geteuid() != required_uid:
        raise AuthorizationError(f"{label} publisher has the wrong effective UID")
    absolute = _lexical_absolute(path, label)
    parent_path = _validate_registry(
        absolute.parent,
        f"{label} directory",
        required_uid=required_uid,
    )
    parent = _open_directory(parent_path, f"{label} directory")
    parent_identity = _directory_anchor(os.fstat(parent))
    try:
        os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as error:
        os.close(parent)
        raise AuthorizationError(f"{label} destination is unsafe") from error
    else:
        os.close(parent)
        raise FileExistsError(absolute)
    temporary = f".{absolute.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, required_uid, os.fstat(descriptor).st_gid)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AuthorizationError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _rename_noreplace_at(
            parent,
            temporary,
            absolute.name,
            label=label,
        )
        published = True
        os.fsync(parent)
        verification_parent = _open_directory(
            parent_path,
            f"{label} directory",
        )
        try:
            if _directory_anchor(os.fstat(verification_parent)) != parent_identity:
                raise AuthorizationError(f"{label} directory changed")
        finally:
            os.close(verification_parent)
        snapshot = _stable_read(
            absolute,
            label,
            required_mode=mode,
            required_uid=required_uid,
            required_nlink=1,
        )
        if snapshot.raw != raw:
            raise AuthorizationError(f"{label} bytes differ after publication")
        return snapshot
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        # Once the destination link exists it is never removed here: publication
        # is the durable consume boundary even if a later verification fails.
        if published:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    label: str,
) -> None:
    """Commit one temporary name without an intermediate hard-link state."""

    if (
        type(parent_descriptor) is not int
        or parent_descriptor < 0
        or not source_name
        or not destination_name
        or "/" in source_name
        or "/" in destination_name
        or "\x00" in source_name
        or "\x00" in destination_name
    ):
        raise AuthorizationError(f"{label} publication names are invalid")
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise AuthorizationError(
            "renameat2(RENAME_NOREPLACE) is unavailable"
        ) from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(destination_name)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        message = "renameat2(RENAME_NOREPLACE) is unsupported"
    else:
        message = f"{label} publication failed"
    raise AuthorizationError(message) from OSError(
        error_number,
        os.strerror(error_number),
    )


def issue_authorization(
    document: dict[str, Any],
    output: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
    source_root: Path | None = None,
) -> AuthorizationRecord:
    validate_authorization_document(
        document,
        now=now,
        required_uid=policy.required_uid,
        policy=policy,
        source_root=source_root,
    )
    raw = canonical_json_bytes(document)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise AuthorizationError("campaign authorization exceeds its size bound")
    snapshot = _publish_no_replace(
        output,
        raw,
        mode=0o444,
        required_uid=policy.required_uid,
        label="campaign authorization",
    )
    return load_authorization(snapshot.path, now=now, policy=policy)


def claim_authorization(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> ClaimRecord:
    authorization = load_authorization(
        authorization_path,
        now=now,
        policy=policy,
        require_fresh_outputs=True,
    )
    registry = _validate_registry(
        policy.claim_registry,
        "campaign claim registry",
        required_uid=policy.required_uid,
    )
    destination = registry / f"{authorization.snapshot.sha256}.claim.json"
    document = {
        "schema_version": CLAIM_SCHEMA,
        "authorization_id": authorization.document["authorization_id"],
        "authorization_path": os.fspath(authorization.snapshot.path),
        "authorization_sha256": authorization.snapshot.sha256,
        "claimed_at": utc_timestamp(now),
        "attempt": 1,
        "max_attempts": 1,
    }
    raw = canonical_json_bytes(document)
    try:
        snapshot = _publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=policy.required_uid,
            label="campaign authorization claim",
        )
    except FileExistsError as error:
        raise AuthorizationConsumed("campaign authorization is already consumed") from error
    return ClaimRecord(snapshot, document, authorization)


def load_claim(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> ClaimRecord:
    authorization = load_authorization(
        authorization_path,
        now=now,
        policy=policy,
        require_fresh_outputs=False,
        enforce_current_window=False,
        require_bound_inputs=False,
    )
    return _load_claim_for_record(authorization, policy=policy)


def load_live_claim(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> ClaimRecord:
    """Load a consumed claim for an in-window campaign execution.

    Unlike :func:`load_claim`, this is not an archival reader.  It revalidates
    the authorization window and all bound input paths on every call.  Outputs
    are allowed to exist because one locked transaction invokes several
    sequential campaign stages under the same one-shot claim.
    """

    authorization = load_authorization(
        authorization_path,
        now=now,
        policy=policy,
        require_fresh_outputs=False,
        enforce_current_window=True,
        require_bound_inputs=True,
    )
    return _load_claim_for_record(authorization, policy=policy)


def publish_outcome(
    claim: ClaimRecord,
    document: dict[str, Any],
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> FileSnapshot:
    """Publish the authorization-derived immutable outcome exactly once."""

    validate_outcome_document(document, claim=claim, policy=policy)
    raw = canonical_json_bytes(document)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise AuthorizationError("campaign outcome exceeds its size bound")
    registry = _validate_registry(
        policy.outcome_registry,
        "campaign outcome registry",
        required_uid=policy.required_uid,
    )
    destination = registry / (
        f"{claim.authorization.snapshot.sha256}.outcome.json"
    )
    try:
        return _publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=policy.required_uid,
            label="campaign outcome",
        )
    except FileExistsError as error:
        raise AuthorizationConsumed(
            "campaign authorization outcome already exists"
        ) from error


def load_outcome(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> tuple[FileSnapshot, dict[str, Any]]:
    """Load and fully bind the immutable outcome to its authorization claim."""

    claim = load_claim(
        authorization_path,
        now=now,
        policy=policy,
    )
    destination = outcome_path(
        claim.authorization.snapshot.sha256,
        policy=policy,
    )
    snapshot = _stable_read(
        destination,
        "campaign outcome",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign outcome")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError("campaign outcome is not canonical JSON")
    validate_outcome_document(document, claim=claim, policy=policy)
    return snapshot, document


def publish_recovery(
    claim: ClaimRecord,
    document: dict[str, Any],
    *,
    policy: RegistryPolicy = RegistryPolicy(),
) -> FileSnapshot:
    """Publish the authorization-derived immutable recovery receipt once."""

    validate_recovery_document(document, claim=claim, policy=policy)
    raw = canonical_json_bytes(document)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise AuthorizationError("campaign recovery receipt exceeds its size bound")
    registry = _validate_registry(
        policy.outcome_registry,
        "campaign outcome registry",
        required_uid=policy.required_uid,
    )
    destination = registry / (
        f"{claim.authorization.snapshot.sha256}.recovery.json"
    )
    try:
        return _publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=policy.required_uid,
            label="campaign recovery receipt",
        )
    except FileExistsError as error:
        raise AuthorizationConsumed(
            "campaign authorization recovery receipt already exists"
        ) from error


def load_recovery(
    authorization_path: Path,
    *,
    now: datetime,
    policy: RegistryPolicy = RegistryPolicy(),
) -> tuple[FileSnapshot, dict[str, Any]]:
    """Load an immutable recovery receipt after its claim window expires."""

    claim = load_claim(authorization_path, now=now, policy=policy)
    destination = recovery_path(
        claim.authorization.snapshot.sha256,
        policy=policy,
    )
    snapshot = _stable_read(
        destination,
        "campaign recovery receipt",
        required_mode=0o444,
        required_uid=policy.required_uid,
        required_nlink=1,
    )
    document = strict_json_bytes(snapshot.raw, "campaign recovery receipt")
    if canonical_json_bytes(document) != snapshot.raw:
        raise AuthorizationError(
            "campaign recovery receipt is not canonical JSON"
        )
    validate_recovery_document(document, claim=claim, policy=policy)
    return snapshot, document


def require_authorization_window_binding(
    record: AuthorizationRecord,
    *,
    source_commit: str,
    source_tree: str,
    aq4_source_root: Path,
    aq4_source_commit: str,
    aq4_source_tree: str,
    before_manifest_sha256: str,
    before_worker_protocol: str,
    before_worker_binary_path: Path,
    before_promotion_receipt_path: Path,
    before_promotion_receipt_sha256: str,
    aq4_promotion_evidence_path: Path,
    aq4_promotion_evidence_sha256: str,
    candidate_manifest_sha256: str,
    candidate_worker_binary_sha256: str,
    candidate_promotion_receipt_sha256: str,
    rollback_backup_path: Path,
) -> None:
    """Bind an operational window to every authorization-owned identity."""

    document = record.document
    source = document["source"]
    before = document["before"]
    aq4_release = document["aq4_release"]
    aq4_source = aq4_release["source"]
    candidate = document["candidate"]
    rollback = document["rollback"]
    if (
        source_commit != source["commit"]
        or source_tree != source["tree"]
        or os.fspath(aq4_source_root) != aq4_source["root"]
        or aq4_source_commit != aq4_source["commit"]
        or aq4_source_tree != aq4_source["tree"]
        or before_manifest_sha256 != before["manifest_sha256"]
        or before_worker_protocol != before["worker_protocol"]
        or os.fspath(before_worker_binary_path)
        != before["worker_binary_path"]
        or os.fspath(before_promotion_receipt_path)
        != before["promotion_receipt_path"]
        or before_promotion_receipt_sha256
        != before["promotion_receipt_sha256"]
        or os.fspath(aq4_promotion_evidence_path)
        != aq4_release["promotion_evidence"]["source_path"]
        or aq4_promotion_evidence_sha256
        != aq4_release["promotion_evidence"]["sha256"]
        or candidate_manifest_sha256 != candidate["manifest_sha256"]
        or candidate_worker_binary_sha256 != candidate["worker_binary_sha256"]
        or candidate_promotion_receipt_sha256
        != candidate["promotion_receipt_sha256"]
        or os.fspath(rollback_backup_path)
        != os.fspath(Path(rollback["backup_path"]))
    ):
        raise AuthorizationError("campaign window identity differs from authorization")


def require_window_binding(
    claim: ClaimRecord,
    *,
    source_commit: str,
    source_tree: str,
    aq4_source_root: Path,
    aq4_source_commit: str,
    aq4_source_tree: str,
    before_manifest_sha256: str,
    before_worker_protocol: str,
    before_worker_binary_path: Path,
    before_promotion_receipt_path: Path,
    before_promotion_receipt_sha256: str,
    aq4_promotion_evidence_path: Path,
    aq4_promotion_evidence_sha256: str,
    candidate_manifest_sha256: str,
    candidate_worker_binary_sha256: str,
    candidate_promotion_receipt_sha256: str,
    rollback_backup_path: Path,
) -> None:
    """Bind a claimed operational window to every authorization-owned identity."""

    require_authorization_window_binding(
        claim.authorization,
        source_commit=source_commit,
        source_tree=source_tree,
        aq4_source_root=aq4_source_root,
        aq4_source_commit=aq4_source_commit,
        aq4_source_tree=aq4_source_tree,
        before_manifest_sha256=before_manifest_sha256,
        before_worker_protocol=before_worker_protocol,
        before_worker_binary_path=before_worker_binary_path,
        before_promotion_receipt_path=before_promotion_receipt_path,
        before_promotion_receipt_sha256=before_promotion_receipt_sha256,
        aq4_promotion_evidence_path=aq4_promotion_evidence_path,
        aq4_promotion_evidence_sha256=aq4_promotion_evidence_sha256,
        candidate_manifest_sha256=candidate_manifest_sha256,
        candidate_worker_binary_sha256=candidate_worker_binary_sha256,
        candidate_promotion_receipt_sha256=candidate_promotion_receipt_sha256,
        rollback_backup_path=rollback_backup_path,
    )


def require_campaign_binding(
    claim: ClaimRecord,
    *,
    campaign_name: str,
    run_id: str,
    final_path: Path,
) -> None:
    """Bind one campaign invocation to its reviewed run and output identity."""

    if campaign_name not in CAMPAIGN_FIELDS:
        raise AuthorizationError("campaign name is not authorized")
    campaign = claim.authorization.document["campaigns"][campaign_name]
    if (
        run_id != campaign["run_id"]
        or os.fspath(final_path) != os.fspath(Path(campaign["final_path"]))
    ):
        raise AuthorizationError("campaign run/output identity differs from authorization")
