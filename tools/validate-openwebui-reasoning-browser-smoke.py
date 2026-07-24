#!/usr/bin/env python3
"""Validate hash-only OpenWebUI reasoning browser smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization  # noqa: E402


SCHEMA_VERSION_V1 = "ullm.openwebui.reasoning_browser_smoke.v1"
SCHEMA_VERSION_V2 = "ullm.openwebui.reasoning_browser_smoke.v2"
SCHEMA_VERSION = SCHEMA_VERSION_V2
SCHEMA_VERSION_V3 = "ullm.openwebui.reasoning_browser_smoke.v3"
SCHEMA_VERSION_V4 = "ullm.openwebui.reasoning_browser_smoke.v4"
SCHEMA_VERSION_V5 = "ullm.openwebui.reasoning_browser_smoke.v5"
VALIDATOR_SCHEMA_VERSION = "ullm.openwebui.reasoning_browser_smoke_validator.v1"
VALIDATOR_SCHEMA_VERSION_V2 = "ullm.openwebui.reasoning_browser_smoke_validator.v2"
VALIDATOR_SCHEMA_VERSION_V3 = "ullm.openwebui.reasoning_browser_smoke_validator.v3"
MAX_EVIDENCE_BYTES = 1 * 1024 * 1024
MAX_PROVIDER_REQUESTS = 4
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z"
)
IMMUTABLE_IMAGE_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*@)?sha256:[0-9a-f]{64}\Z"
)
FORBIDDEN_KEYS = {
    "prompt",
    "response",
    "content",
    "request_body",
    "response_body",
    "authorization",
    "api_key",
    "token",
    "conversation",
    "raw",
    "screenshot",
}
SWITCH_EVIDENCE_FIELDS = {
    "provider_switch_performed",
    "provider_switch_model_id_sha256",
    "provider_switch_answer",
    "provider_return_performed",
    "provider_return_model_id_sha256",
    "provider_return_answer",
}
CAMPAIGN_LINEAGE_SCHEMA_V2 = "ullm.served_model.campaign_lineage.v2"
ACTIVE_BINDING_SCHEMA = "ullm.served_model.active_binding.v1"
ACTIVE_OBSERVATION_SCHEMA = "ullm.served_model.active_manifest_observation.v1"
BROWSER_EVIDENCE_FILE = "browser-evidence.json"
BROWSER_LINEAGE_ARTIFACTS = frozenset(
    {
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)
BROWSER_OUTPUT_FILES_V2 = frozenset(
    {*BROWSER_LINEAGE_ARTIFACTS, BROWSER_EVIDENCE_FILE}
)
ACTIVE_BINDING_STAGES = (
    "preflight",
    "browser-launch",
    "browser-complete",
    "validation",
    "publication",
)


class ValidationError(ValueError):
    """Raised when browser evidence violates the hash-only contract."""


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_read(
    path: Path,
    label: str,
    maximum: int,
    *,
    require_immutable: bool = False,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValidationError("O_NOFOLLOW is required for browser validation")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError(f"{label} must be a regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
            or (
                require_immutable
                and (
                    stat.S_IMODE(before.st_mode) != 0o444
                    or before.st_nlink != 1
                )
            )
        ):
            raise ValidationError(f"{label} identity differs")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        named = path.lstat()
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(named)
        ):
            raise ValidationError(f"{label} changed while being read")
        return bytes(raw)
    except OSError as error:
        raise ValidationError(f"{label} could not be read") from error
    finally:
        os.close(descriptor)


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} is not strict JSON") from error
    return value


def _load(path: Path) -> dict[str, Any]:
    value = _decode_json(
        _stable_read(path, "browser evidence", MAX_EVIDENCE_BYTES),
        "browser evidence",
    )
    if not isinstance(value, dict):
        raise ValidationError("browser evidence root is not an object")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("browser evidence contains duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValidationError("browser evidence contains a non-finite number")


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValidationError(f"browser evidence contains forbidden field: {key}")
            _scan_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden(child)


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a lowercase SHA-256")


def _identity(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "manifest_sha256",
        "worker_binary_sha256",
        "tokenizer_sha256",
        "openwebui_image",
    }:
        raise ValidationError("identity fields differ")
    for field in ("manifest_sha256", "worker_binary_sha256", "tokenizer_sha256"):
        _hash(value[field], f"identity.{field}")
    if (
        not isinstance(value["openwebui_image"], str)
        or IMAGE_RE.fullmatch(value["openwebui_image"]) is None
    ):
        raise ValidationError("identity.openwebui_image is not content-addressed")


def _openwebui_server(value: Any, identity: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"before", "after"}:
        raise ValidationError("openwebui_server fields differ")
    expected_fields = {
        "container_id",
        "image_id",
        "config_image",
        "name",
        "running",
        "pid",
        "started_at",
    }
    for label in ("before", "after"):
        observation = value[label]
        if (
            not isinstance(observation, dict)
            or set(observation) != expected_fields
            or not isinstance(observation["container_id"], str)
            or re.fullmatch(r"[0-9a-f]{64}", observation["container_id"])
            is None
            or not isinstance(observation["image_id"], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", observation["image_id"])
            is None
            or not isinstance(observation["config_image"], str)
            or not observation["config_image"]
            or len(observation["config_image"].encode("utf-8")) > 1024
            or not isinstance(observation["name"], str)
            or not observation["name"].startswith("/")
            or len(observation["name"].encode("utf-8")) > 256
            or observation["running"] is not True
            or type(observation["pid"]) is not int
            or observation["pid"] <= 0
            or not isinstance(observation["started_at"], str)
            or not observation["started_at"]
            or len(observation["started_at"]) > 128
            or not observation["started_at"].isascii()
        ):
            raise ValidationError(f"openwebui_server.{label} differs")
    if value["before"] != value["after"]:
        raise ValidationError("OpenWebUI server identity changed during browser gate")
    expected_static = {
        "image_id": authorization.FIXED_OPENWEBUI_IMAGE.rsplit("@", 1)[1],
        "config_image": authorization.FIXED_OPENWEBUI_CONFIG_IMAGE,
        "name": f"/{authorization.FIXED_OPENWEBUI_CONTAINER_NAME}",
        "running": True,
    }
    if (
        identity.get("openwebui_image") != authorization.FIXED_OPENWEBUI_IMAGE
        or any(
            {
                field: value[label][field]
                for field in expected_static
            }
            != expected_static
            for label in ("before", "after")
        )
    ):
        raise ValidationError(
            "OpenWebUI server observation differs from fixed identity"
        )


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValidationError(f"{label} is invalid")


def _text_evidence(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"utf8_bytes", "sha256"}:
        raise ValidationError(f"{label} fields differ")
    _integer(value["utf8_bytes"], f"{label}.utf8_bytes", minimum=1, maximum=1_000_000)
    _hash(value["sha256"], f"{label}.sha256")


def _request(value: Any, index: int, *, version: str) -> None:
    expected = {
        "sha256",
        "utf8_bytes",
        "has_reasoning_content_key",
        "assistant_has_reasoning_content",
    }
    if version in {
        SCHEMA_VERSION_V2,
        SCHEMA_VERSION_V3,
        SCHEMA_VERSION_V4,
        SCHEMA_VERSION_V5,
    }:
        expected.add("model_id_sha256")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValidationError(f"provider request {index} fields differ")
    _hash(value["sha256"], f"provider request {index}.sha256")
    _integer(value["utf8_bytes"], f"provider request {index}.utf8_bytes", minimum=2)
    if version in {
        SCHEMA_VERSION_V2,
        SCHEMA_VERSION_V3,
        SCHEMA_VERSION_V4,
        SCHEMA_VERSION_V5,
    }:
        _hash(value["model_id_sha256"], f"provider request {index}.model_id_sha256")
    if type(value["has_reasoning_content_key"]) is not bool or type(
        value["assistant_has_reasoning_content"]
    ) is not bool:
        raise ValidationError(f"provider request {index} flags are invalid")


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{label} fields differ")
    return value


def _lineage_inventory(artifacts: dict[str, Any]) -> str:
    canonical = json.dumps(
        artifacts,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(
        CAMPAIGN_LINEAGE_SCHEMA_V2.encode("ascii") + b"\0" + canonical
    ).hexdigest()


def _validate_campaign_lineage(
    path: Path,
    value: Any,
    document: dict[str, Any],
    *,
    lineage_root_override: Path | None,
) -> dict[str, Any]:
    lineage = _exact_object(
        value,
        {
            "schema_version",
            "campaign",
            "claim",
            "artifacts",
            "artifact_inventory_sha256",
            "observations",
        },
        "browser campaign lineage",
    )
    if lineage["schema_version"] != CAMPAIGN_LINEAGE_SCHEMA_V2:
        raise ValidationError("browser campaign lineage schema differs")
    campaign = _exact_object(
        lineage["campaign"],
        {"name", "run_id", "final_path", "final_kind", "files"},
        "browser campaign identity",
    )
    if (
        campaign["name"] != "reasoning_browser"
        or not isinstance(campaign["run_id"], str)
        or not campaign["run_id"]
        or campaign["final_kind"] != "directory"
        or campaign["files"] != sorted(BROWSER_OUTPUT_FILES_V2)
        or not isinstance(campaign["final_path"], str)
    ):
        raise ValidationError("browser campaign identity differs")
    final_path = Path(campaign["final_path"])
    if not final_path.is_absolute() or final_path != final_path.absolute():
        raise ValidationError("browser campaign final path is not canonical")
    if lineage_root_override is None:
        try:
            root = final_path.resolve(strict=True)
        except OSError as error:
            raise ValidationError("browser campaign final output is unavailable") from error
        if root != final_path or path.resolve(strict=True) != root / BROWSER_EVIDENCE_FILE:
            raise ValidationError("browser evidence is not its authorized final output")
    else:
        try:
            root = lineage_root_override.resolve(strict=True)
        except OSError as error:
            raise ValidationError("browser staged lineage root is unavailable") from error
        if path.resolve(strict=True) != root / BROWSER_EVIDENCE_FILE:
            raise ValidationError("browser staged evidence location differs")
    try:
        root_metadata = root.lstat()
        observed_names = {entry.name for entry in os.scandir(root)}
    except OSError as error:
        raise ValidationError("browser campaign output cannot be enumerated") from error
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
        or observed_names != BROWSER_OUTPUT_FILES_V2
    ):
        raise ValidationError("browser campaign output layout differs")
    if (
        _decode_json(
            _stable_read(
                path,
                "browser evidence",
                MAX_EVIDENCE_BYTES,
                require_immutable=True,
            ),
            "browser evidence",
        )
        != document
    ):
        raise ValidationError("browser evidence changed during validation")
    identity = document["identity"]

    artifacts = lineage["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != BROWSER_LINEAGE_ARTIFACTS:
        raise ValidationError("browser lineage artifact set differs")
    raws: dict[str, bytes] = {}
    for name in sorted(BROWSER_LINEAGE_ARTIFACTS):
        reference = _exact_object(
            artifacts[name],
            {"bytes", "sha256"},
            f"browser lineage artifact {name}",
        )
        _integer(
            reference["bytes"],
            f"browser lineage artifact {name}.bytes",
            minimum=1,
            maximum=MAX_EVIDENCE_BYTES,
        )
        _hash(reference["sha256"], f"browser lineage artifact {name}.sha256")
        raw = _stable_read(
            root / name,
            f"browser lineage artifact {name}",
            MAX_EVIDENCE_BYTES,
            require_immutable=True,
        )
        if (
            len(raw) != reference["bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["sha256"]
        ):
            raise ValidationError(f"browser lineage artifact {name} differs")
        raws[name] = raw
    _hash(
        lineage["artifact_inventory_sha256"],
        "browser lineage artifact_inventory_sha256",
    )
    inventory = _lineage_inventory(artifacts)
    if lineage["artifact_inventory_sha256"] != inventory:
        raise ValidationError("browser lineage artifact inventory differs")

    claim = _exact_object(
        lineage["claim"],
        {
            "path",
            "sha256",
            "bytes",
            "authorization_path",
            "authorization_sha256",
        },
        "browser campaign claim",
    )
    for field in ("sha256", "authorization_sha256"):
        _hash(claim[field], f"browser campaign claim.{field}")
    _integer(
        claim["bytes"],
        "browser campaign claim.bytes",
        minimum=1,
        maximum=MAX_EVIDENCE_BYTES,
    )
    for field in ("path", "authorization_path"):
        if (
            not isinstance(claim[field], str)
            or not Path(claim[field]).is_absolute()
            or Path(claim[field]) != Path(claim[field]).absolute()
        ):
            raise ValidationError(f"browser campaign claim.{field} differs")
    claim_raw = _stable_read(
        Path(claim["path"]),
        "browser campaign claim",
        MAX_EVIDENCE_BYTES,
        require_immutable=True,
    )
    authorization_raw = _stable_read(
        Path(claim["authorization_path"]),
        "browser campaign authorization",
        MAX_EVIDENCE_BYTES,
        require_immutable=True,
    )
    if (
        len(claim_raw) != claim["bytes"]
        or hashlib.sha256(claim_raw).hexdigest() != claim["sha256"]
        or hashlib.sha256(authorization_raw).hexdigest()
        != claim["authorization_sha256"]
    ):
        raise ValidationError("browser campaign authorization bytes differ")

    binding = _decode_json(
        raws["active-manifest-binding.json"],
        "browser active binding",
    )
    _exact_object(
        binding,
        {
            "schema_version",
            "status",
            "candidate",
            "actual_active_path",
            "expected_stages",
            "observation_count",
            "observations",
            "claim",
            "campaign",
        },
        "browser active binding",
    )
    candidate = _exact_object(
        binding["candidate"],
        {"artifact", "source_path", "sha256", "bytes"},
        "browser binding candidate",
    )
    observations_reference = _exact_object(
        binding["observations"],
        {"artifact", "sha256", "bytes"},
        "browser binding observations",
    )
    binding_campaign = _exact_object(
        binding["campaign"],
        {"name", "run_id", "final_path"},
        "browser binding campaign",
    )
    candidate_raw = raws["candidate-served-model.json"]
    observations_raw = raws["active-manifest-observations.jsonl"]
    if (
        binding["schema_version"] != ACTIVE_BINDING_SCHEMA
        or binding["status"] != "complete"
        or binding["expected_stages"] != list(ACTIVE_BINDING_STAGES)
        or binding["observation_count"] != len(ACTIVE_BINDING_STAGES)
        or binding["claim"] != claim
        or binding_campaign
        != {
            "name": "reasoning_browser",
            "run_id": campaign["run_id"],
            "final_path": campaign["final_path"],
        }
        or candidate["artifact"] != "candidate-served-model.json"
        or candidate["sha256"] != hashlib.sha256(candidate_raw).hexdigest()
        or candidate["sha256"] != identity["manifest_sha256"]
        or candidate["bytes"] != len(candidate_raw)
        or observations_reference
        != {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": hashlib.sha256(observations_raw).hexdigest(),
            "bytes": len(observations_raw),
        }
    ):
        raise ValidationError("browser active binding differs")

    observations = _exact_object(
        lineage["observations"],
        {"count", "stages"},
        "browser observation lineage",
    )
    lines = observations_raw.splitlines(keepends=True)
    if (
        observations["count"] != len(ACTIVE_BINDING_STAGES)
        or not isinstance(observations["stages"], list)
        or len(observations["stages"]) != len(ACTIVE_BINDING_STAGES)
        or len(lines) != len(ACTIVE_BINDING_STAGES)
        or any(not line.endswith(b"\n") for line in lines)
    ):
        raise ValidationError("browser observation count differs")
    for sequence, (stage, line, expected_stage) in enumerate(
        zip(
            observations["stages"],
            lines,
            ACTIVE_BINDING_STAGES,
            strict=True,
        )
    ):
        stage = _exact_object(
            stage,
            {"sequence", "stage", "sha256"},
            f"browser observation lineage {sequence}",
        )
        row = _decode_json(line, f"browser observation {sequence}")
        _exact_object(
            row,
            {
                "schema_version",
                "sequence",
                "stage",
                "observed_unix_ns",
                "observed_monotonic_ns",
                "candidate",
                "active",
                "bytes_equal",
                "claim",
            },
            f"browser observation {sequence}",
        )
        candidate_row = _exact_object(
            row["candidate"],
            {"path", "sha256", "identity"},
            f"browser observation {sequence} candidate",
        )
        active_row = _exact_object(
            row["active"],
            {"path", "sha256", "identity"},
            f"browser observation {sequence} active",
        )
        if (
            stage
            != {
                "sequence": sequence,
                "stage": expected_stage,
                "sha256": hashlib.sha256(line).hexdigest(),
            }
            or row["schema_version"] != ACTIVE_OBSERVATION_SCHEMA
            or row["sequence"] != sequence
            or row["stage"] != expected_stage
            or row["bytes_equal"] is not True
            or row["claim"] != claim
            or candidate_row["path"] != candidate["source_path"]
            or candidate_row["sha256"] != candidate["sha256"]
            or active_row["path"] != binding["actual_active_path"]
            or active_row["sha256"] != candidate["sha256"]
        ):
            raise ValidationError(f"browser observation {sequence} differs")
        for label, file_row in (("candidate", candidate_row), ("active", active_row)):
            file_identity = _exact_object(
                file_row["identity"],
                {
                    "device",
                    "inode",
                    "mode",
                    "links",
                    "uid",
                    "gid",
                    "bytes",
                    "mtime_ns",
                    "ctime_ns",
                },
                f"browser observation {sequence} {label} identity",
            )
            if (
                any(
                    type(item) is not int or item < 0
                    for item in file_identity.values()
                )
                or file_identity["bytes"] != candidate["bytes"]
            ):
                raise ValidationError(
                    f"browser observation {sequence} identity differs"
                )
    return {
        "schema_version": CAMPAIGN_LINEAGE_SCHEMA_V2,
        "campaign_name": campaign["name"],
        "run_id": campaign["run_id"],
        "final_path": campaign["final_path"],
        "claim_sha256": claim["sha256"],
        "authorization_sha256": claim["authorization_sha256"],
        "candidate_sha256": candidate["sha256"],
        "observation_count": observations["count"],
        "artifact_inventory_sha256": inventory,
    }


def validate(
    path: Path,
    *,
    lineage_root_override: Path | None = None,
) -> dict[str, Any]:
    document = _load(path)
    _scan_forbidden(document)
    expected_v1 = {
        "schema_version",
        "model_id_sha256",
        "first_answer",
        "expanded_view",
        "second_answer",
        "reasoning_details_expanded",
        "provider_request_count",
        "provider_requests",
        "hidden_reasoning_reinserted",
        "page_error_count",
        "page_error_digests",
    }
    version = document.get("schema_version")
    switch_cycle = False
    if version == SCHEMA_VERSION_V1:
        expected = expected_v1
    elif version in {
        SCHEMA_VERSION_V2,
        SCHEMA_VERSION_V3,
        SCHEMA_VERSION_V4,
        SCHEMA_VERSION_V5,
    }:
        switch_fields = set(document) & SWITCH_EVIDENCE_FIELDS
        if switch_fields and switch_fields != SWITCH_EVIDENCE_FIELDS:
            raise ValidationError("browser evidence switch fields differ")
        switch_cycle = bool(switch_fields)
        expected = expected_v1 | switch_fields
        if version in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4, SCHEMA_VERSION_V5}:
            expected |= {"source_commit", "identity"}
        if version in {SCHEMA_VERSION_V4, SCHEMA_VERSION_V5}:
            expected.add("campaign_lineage")
        if version == SCHEMA_VERSION_V5:
            expected |= {"browser_image", "openwebui_server"}
    else:
        expected = set()
    if set(document) != expected:
        raise ValidationError("browser evidence root fields differ")
    if version in {SCHEMA_VERSION_V3, SCHEMA_VERSION_V4, SCHEMA_VERSION_V5}:
        source_commit = document["source_commit"]
        if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
            raise ValidationError("source_commit is not a full lowercase Git commit")
        _identity(document["identity"])
    _hash(document["model_id_sha256"], "model_id_sha256")
    _text_evidence(document["first_answer"], "first_answer")
    _text_evidence(document["expanded_view"], "expanded_view")
    if document["expanded_view"]["utf8_bytes"] <= document["first_answer"]["utf8_bytes"]:
        raise ValidationError("expanded view has no additional visible details")
    _text_evidence(document["second_answer"], "second_answer")
    if document["reasoning_details_expanded"] is not True:
        raise ValidationError("reasoning details were not expanded")
    _integer(
        document["provider_request_count"],
        "provider_request_count",
        minimum=2,
        maximum=MAX_PROVIDER_REQUESTS,
    )
    requests = document["provider_requests"]
    if not isinstance(requests, list) or len(requests) != document["provider_request_count"]:
        raise ValidationError("provider request count differs")
    for index, request in enumerate(requests):
        _request(request, index, version=version)
    if version in {
        SCHEMA_VERSION_V2,
        SCHEMA_VERSION_V3,
        SCHEMA_VERSION_V4,
        SCHEMA_VERSION_V5,
    }:
        if switch_cycle:
            if document["provider_switch_performed"] is not True:
                raise ValidationError("provider switch was not performed")
            _hash(
                document["provider_switch_model_id_sha256"],
                "provider_switch_model_id_sha256",
            )
            _text_evidence(document["provider_switch_answer"], "provider_switch_answer")
            if document["provider_switch_model_id_sha256"] == document["model_id_sha256"]:
                raise ValidationError("provider switch model is not distinct")
            if document["provider_return_performed"] is not True:
                raise ValidationError("provider return was not performed")
            _hash(
                document["provider_return_model_id_sha256"],
                "provider_return_model_id_sha256",
            )
            _text_evidence(document["provider_return_answer"], "provider_return_answer")
            if len(requests) < 4:
                raise ValidationError("provider switch request is missing")
            if any(
                request["model_id_sha256"] != document["model_id_sha256"]
                for request in requests[:2]
            ):
                raise ValidationError("initial provider request model differs")
            if (
                requests[-2]["model_id_sha256"]
                != document["provider_switch_model_id_sha256"]
            ):
                raise ValidationError("provider switch request model differs")
            if requests[-1]["model_id_sha256"] == requests[-2]["model_id_sha256"]:
                raise ValidationError("provider return request model is not distinct")
            if (
                requests[-1]["model_id_sha256"]
                != document["provider_return_model_id_sha256"]
            ):
                raise ValidationError("provider return request model differs")
        else:
            if len(requests) != 2:
                raise ValidationError("v2 no-switch evidence must contain two provider requests")
            if any(
                request["model_id_sha256"] != document["model_id_sha256"]
                for request in requests
            ):
                raise ValidationError("v2 no-switch provider request model differs")
    if document["hidden_reasoning_reinserted"] is not False:
        raise ValidationError("hidden reasoning was reinserted")
    _integer(document["page_error_count"], "page_error_count", maximum=0)
    page_errors = document["page_error_digests"]
    if not isinstance(page_errors, list) or page_errors:
        raise ValidationError("page error digests are not empty")
    reasons: list[str] = []
    if requests[-1]["assistant_has_reasoning_content"]:
        reasons.append("last provider request contains assistant reasoning_content")
    campaign_lineage = (
        _validate_campaign_lineage(
            path,
            document["campaign_lineage"],
            document,
            lineage_root_override=lineage_root_override,
        )
        if version in {SCHEMA_VERSION_V4, SCHEMA_VERSION_V5}
        else None
    )
    if lineage_root_override is not None and version not in {
        SCHEMA_VERSION_V4,
        SCHEMA_VERSION_V5,
    }:
        raise ValidationError(
            "lineage root override requires lineage-bearing browser evidence"
        )
    if version == SCHEMA_VERSION_V5 and (
        not isinstance(document["browser_image"], str)
        or IMMUTABLE_IMAGE_RE.fullmatch(document["browser_image"]) is None
    ):
        raise ValidationError("browser_image is not content-addressed")
    if version == SCHEMA_VERSION_V5:
        _openwebui_server(
            document["openwebui_server"],
            document["identity"],
        )
    return {
        "schema_version": (
            VALIDATOR_SCHEMA_VERSION_V3
            if version == SCHEMA_VERSION_V5
            else (
                VALIDATOR_SCHEMA_VERSION_V2
                if version == SCHEMA_VERSION_V4
                else VALIDATOR_SCHEMA_VERSION
            )
        ),
        "input_schema_version": version,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "provider_request_count": len(requests),
        "reasons": reasons,
        **(
            {}
            if campaign_lineage is None
            else {"campaign_lineage": campaign_lineage}
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.evidence)
    except Exception as error:
        print(f"OpenWebUI reasoning browser validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["gate_eligible"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
