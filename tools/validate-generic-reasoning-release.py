#!/usr/bin/env python3
"""Validate bounded, hash-only generic reasoning release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION_V1 = "ullm.generic_reasoning_release_evidence.v1"
SCHEMA_VERSION_V2 = "ullm.generic_reasoning_release_evidence.v2"
SCHEMA_VERSION = SCHEMA_VERSION_V1
VALIDATOR_SCHEMA_VERSION = "ullm.generic_reasoning_release_validator.v1"
VALIDATOR_SCHEMA_VERSION_V2 = "ullm.generic_reasoning_release_validator.v2"
REQUIRED_MODES = {"disabled", "budget-32", "budget-128", "budget-256", "unbounded"}
HASH_FIELDS = {"manifest_sha256", "worker_binary_sha256", "tokenizer_sha256", "prompt_sha256"}
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_DIGEST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z")
FORBIDDEN_KEYS = {
    "prompt",
    "response",
    "request_body",
    "response_body",
    "authorization",
    "api_key",
    "token",
    "conversation",
}
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_CASES = 4096
MAX_SSE_CHUNKS = 1_000_000
MAX_LIFECYCLE_EVENTS = 4096
LIFECYCLE_SCHEMA_VERSION = "ullm.generic_reasoning_lifecycle_evidence.v1"
CAMPAIGN_LINEAGE_SCHEMA_V2 = "ullm.served_model.campaign_lineage.v2"
ACTIVE_BINDING_SCHEMA = "ullm.served_model.active_binding.v1"
ACTIVE_OBSERVATION_SCHEMA = "ullm.served_model.active_manifest_observation.v1"
FIXED_ACTIVE_MANIFEST_PATH = "/etc/ullm/served-models/active.json"
REASONING_CAMPAIGN_SCHEMA_V2 = "ullm.generic_reasoning_release_campaign.v2"
REASONING_CAMPAIGN_STAGES = (
    "preflight",
    *(
        stage
        for mode in ("disabled", "budget-32", "budget-128", "budget-256", "unbounded")
        for stage in (f"{mode}:stream", f"{mode}:nonstream")
    ),
    "final",
)
REASONING_CAMPAIGN_FILES = frozenset(
    {
        "cases.json",
        "lifecycle.json",
        "resource-samples.jsonl",
        "summary.json",
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)


class ValidationError(ValueError):
    """Raised when release evidence violates the published contract."""


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
        raise ValidationError("O_NOFOLLOW is required for release validation")
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
            if before.st_size > maximum:
                raise ValidationError(f"{label} exceeds its size bound")
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
        try:
            named = path.lstat()
        except OSError as error:
            raise ValidationError(f"{label} disappeared while being read") from error
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(named)
        ):
            raise ValidationError(f"{label} changed while being read")
        return bytes(raw)
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
        raise ValidationError(f"{label} is not valid JSON") from error
    return value


def _load(path: Path) -> dict[str, Any]:
    value = _decode_json(
        _stable_read(path, "release evidence", MAX_EVIDENCE_BYTES),
        "release evidence",
    )
    if not isinstance(value, dict):
        raise ValidationError("release evidence root must be an object")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("release evidence contains duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValidationError("release evidence contains a non-finite number")


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValidationError(f"release evidence contains forbidden field: {key}")
            _scan_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden(child)


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValidationError(f"{label} is not a lowercase SHA-256")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 512:
        raise ValidationError(f"{label} is invalid")


def _commit(value: Any, label: str) -> None:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a lowercase Git commit")


def _image_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or IMAGE_DIGEST_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a content-addressed image")


def _integer(value: Any, label: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} is invalid")


def _number(value: Any, label: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is invalid")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise ValidationError(f"{label} is invalid")


def _percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValidationError("percentile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _validate_identity(identity: Any) -> None:
    if not isinstance(identity, dict):
        raise ValidationError("release evidence identity is missing")
    if set(identity) != {
        "manifest_sha256",
        "worker_binary_sha256",
        "tokenizer_sha256",
        "openwebui_image",
    }:
        raise ValidationError("release evidence identity fields differ")
    for field in HASH_FIELDS - {"prompt_sha256"}:
        _hash(identity[field], f"identity.{field}")
    _image_digest(identity["openwebui_image"], "identity.openwebui_image")


def _validate_case(case: Any) -> str:
    if not isinstance(case, dict):
        raise ValidationError("release evidence case is not an object")
    expected = {
        "id",
        "mode",
        "prompt_fixture_id",
        "prompt_sha256",
        "stream",
        "http_status",
        "sse_chunk_count",
        "finish_reason",
        "raw",
        "timing",
        "resource",
        "quality",
    }
    if set(case) != expected:
        raise ValidationError("release evidence case fields differ")
    _text(case["id"], "case.id")
    mode = case["mode"]
    if mode not in REQUIRED_MODES:
        raise ValidationError("release evidence case mode is invalid")
    _text(case["prompt_fixture_id"], "case.prompt_fixture_id")
    _hash(case["prompt_sha256"], "case.prompt_sha256")
    if type(case["stream"]) is not bool or case["http_status"] != 200:
        raise ValidationError("release evidence HTTP contract failed")
    _integer(case["sse_chunk_count"], "case.sse_chunk_count")
    if case["sse_chunk_count"] > MAX_SSE_CHUNKS:
        raise ValidationError("case SSE chunk count exceeds its bound")
    if case["stream"] and case["sse_chunk_count"] < 1:
        raise ValidationError("stream case has no SSE chunks")
    if not case["stream"] and case["sse_chunk_count"] != 0:
        raise ValidationError("non-stream case has SSE chunks")
    if case["finish_reason"] not in {"stop", "length"}:
        raise ValidationError("release evidence finish reason is invalid")

    raw = case["raw"]
    if not isinstance(raw, dict) or set(raw) != {
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "forced_end_tokens",
        "answer_tokens",
        "budget_overshoot",
        "empty_answer",
        "usage_completion_tokens",
    }:
        raise ValidationError("release evidence raw metrics differ")
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "forced_end_tokens",
        "answer_tokens",
        "budget_overshoot",
        "usage_completion_tokens",
    ):
        _integer(raw[field], f"case.raw.{field}")
    if type(raw["empty_answer"]) is not bool:
        raise ValidationError("case.raw.empty_answer is invalid")
    if raw["completion_tokens"] != raw["reasoning_tokens"] + raw["forced_end_tokens"] + raw["answer_tokens"]:
        raise ValidationError("case raw token accounting differs")
    if raw["usage_completion_tokens"] != raw["completion_tokens"]:
        raise ValidationError("case usage completion count differs")
    if raw["budget_overshoot"] != 0:
        raise ValidationError("case budget overshoot is nonzero")
    if raw["empty_answer"] or raw["answer_tokens"] < 1:
        raise ValidationError("case has an empty answer")
    if mode == "disabled" and (raw["reasoning_tokens"] or raw["forced_end_tokens"]):
        raise ValidationError("disabled case contains reasoning tokens")
    mode_budget = {
        "budget-32": 32,
        "budget-128": 128,
        "budget-256": 256,
    }.get(mode)
    if mode_budget is not None and raw["reasoning_tokens"] > mode_budget:
        raise ValidationError("case reasoning tokens exceed requested budget")

    timing = case["timing"]
    if not isinstance(timing, dict) or set(timing) != {
        "prefill_tokens_per_second",
        "first_reasoning_token_ms",
        "first_answer_token_ms",
        "reasoning_decode_tokens_per_second",
        "answer_decode_tokens_per_second",
        "decode_tokens_per_second",
        "latency_ms",
    }:
        raise ValidationError("release evidence timing fields differ")
    for field, value in timing.items():
        if value is not None:
            _number(value, f"case.timing.{field}")

    resource = case["resource"]
    if not isinstance(resource, dict) or set(resource) != {
        "rss_delta_bytes",
        "vram_delta_bytes",
        "gpu_temperature_c",
        "power_w",
    }:
        raise ValidationError("release evidence resource fields differ")
    for field, value in resource.items():
        _number(value, f"case.resource.{field}")

    quality = case["quality"]
    if not isinstance(quality, dict) or set(quality) != {"correct", "score"}:
        raise ValidationError("release evidence quality fields differ")
    if type(quality["correct"]) is not bool:
        raise ValidationError("case.quality.correct is invalid")
    _number(quality["score"], "case.quality.score")
    if float(quality["score"]) > 1.0:
        raise ValidationError("case.quality.score exceeds one")
    return mode


def _validate_lifecycle(value: Any, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "events"}:
        raise ValidationError("release evidence lifecycle fields differ")
    if value["schema_version"] != LIFECYCLE_SCHEMA_VERSION:
        raise ValidationError("release evidence lifecycle schema differs")
    events = value["events"]
    if not isinstance(events, list) or len(events) > MAX_LIFECYCLE_EVENTS:
        raise ValidationError("release evidence lifecycle events are invalid")
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {
            "case_id",
            "stream",
            "outcome",
            "prompt_tokens",
            "completion_tokens",
            "reset_complete",
            "reasoning_tokens",
            "forced_end_tokens",
            "admit_to_start_ns",
            "start_to_release_ns",
            "admit_to_release_ns",
        }:
            raise ValidationError("release evidence lifecycle event fields differ")
        _text(event["case_id"], "lifecycle.case_id")
        case_id = event["case_id"]
        if case_id in seen:
            raise ValidationError("release evidence lifecycle case IDs are duplicated")
        seen.add(case_id)
        case = cases.get(case_id)
        if case is None:
            raise ValidationError("release evidence lifecycle case is unknown")
        if type(event["stream"]) is not bool or event["stream"] != case["stream"]:
            raise ValidationError("release evidence lifecycle stream differs")
        if event["outcome"] != case["finish_reason"]:
            raise ValidationError("release evidence lifecycle outcome differs")
        for field, case_value in (
            ("prompt_tokens", case["raw"]["prompt_tokens"]),
            ("completion_tokens", case["raw"]["completion_tokens"]),
        ):
            _integer(event[field], f"lifecycle.{field}")
            if event[field] != case_value:
                raise ValidationError(f"release evidence lifecycle {field} differs")
        if event["reset_complete"] is not True:
            raise ValidationError("release evidence lifecycle reset is incomplete")
        for field in ("reasoning_tokens", "forced_end_tokens"):
            accounting = event[field]
            if accounting is not None:
                _integer(accounting, f"lifecycle.{field}")
        if case["mode"] == "disabled":
            if event["reasoning_tokens"] is not None or event["forced_end_tokens"] is not None:
                raise ValidationError("disabled lifecycle event contains reasoning accounting")
        else:
            if (
                event["reasoning_tokens"] != case["raw"]["reasoning_tokens"]
                or event["forced_end_tokens"] != case["raw"]["forced_end_tokens"]
            ):
                raise ValidationError("release evidence lifecycle accounting differs")
        for field in (
            "admit_to_start_ns",
            "start_to_release_ns",
            "admit_to_release_ns",
        ):
            _integer(event[field], f"lifecycle.{field}")
    return {"event_count": len(events), "case_ids": seen}


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{label} fields differ")
    return value


def _artifact_inventory_sha256(artifacts: dict[str, Any]) -> str:
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
    value: Any,
    *,
    cases: list[dict[str, Any]],
    lifecycle: dict[str, Any],
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
        "release campaign lineage",
    )
    if lineage["schema_version"] != CAMPAIGN_LINEAGE_SCHEMA_V2:
        raise ValidationError("release campaign lineage schema differs")
    campaign = _exact_object(
        lineage["campaign"],
        {"name", "run_id", "final_path", "final_kind", "files"},
        "release campaign identity",
    )
    if (
        campaign["name"] != "reasoning_release"
        or not isinstance(campaign["run_id"], str)
        or not campaign["run_id"]
        or campaign["final_kind"] != "directory"
        or campaign["files"] != sorted(REASONING_CAMPAIGN_FILES)
        or not isinstance(campaign["final_path"], str)
    ):
        raise ValidationError("release campaign identity differs")
    root_path = Path(campaign["final_path"])
    if not root_path.is_absolute() or root_path != root_path.absolute():
        raise ValidationError("release campaign final path is not canonical")
    try:
        root = root_path.resolve(strict=True)
        metadata = root.lstat()
        observed_names = {entry.name for entry in os.scandir(root)}
    except OSError as error:
        raise ValidationError("release campaign final output is unavailable") from error
    if (
        root != root_path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or observed_names != REASONING_CAMPAIGN_FILES
    ):
        raise ValidationError("release campaign final output layout differs")

    artifacts = lineage["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != REASONING_CAMPAIGN_FILES:
        raise ValidationError("release campaign artifact set differs")
    raws: dict[str, bytes] = {}
    for name in sorted(REASONING_CAMPAIGN_FILES):
        reference = _exact_object(
            artifacts[name],
            {"bytes", "sha256"},
            f"release campaign artifact {name}",
        )
        _integer(reference["bytes"], f"release campaign artifact {name}.bytes", minimum=1)
        _hash(reference["sha256"], f"release campaign artifact {name}.sha256")
        raw = _stable_read(
            root / name,
            f"release campaign artifact {name}",
            MAX_EVIDENCE_BYTES,
            require_immutable=True,
        )
        if (
            len(raw) != reference["bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["sha256"]
        ):
            raise ValidationError(f"release campaign artifact {name} differs")
        raws[name] = raw
    _hash(
        lineage["artifact_inventory_sha256"],
        "release campaign artifact_inventory_sha256",
    )
    inventory = _artifact_inventory_sha256(artifacts)
    if lineage["artifact_inventory_sha256"] != inventory:
        raise ValidationError("release campaign artifact inventory differs")

    if _decode_json(raws["cases.json"], "release campaign cases") != cases:
        raise ValidationError("release campaign cases differ from release evidence")
    if (
        _decode_json(raws["lifecycle.json"], "release campaign lifecycle")
        != lifecycle
    ):
        raise ValidationError("release campaign lifecycle differs from release evidence")

    claim = _exact_object(
        lineage["claim"],
        {
            "path",
            "sha256",
            "bytes",
            "authorization_path",
            "authorization_sha256",
        },
        "release campaign claim",
    )
    for field in ("sha256", "authorization_sha256"):
        _hash(claim[field], f"release campaign claim.{field}")
    _integer(claim["bytes"], "release campaign claim.bytes", minimum=1)
    for field in ("path", "authorization_path"):
        if (
            not isinstance(claim[field], str)
            or not Path(claim[field]).is_absolute()
            or Path(claim[field]) != Path(claim[field]).absolute()
        ):
            raise ValidationError(f"release campaign claim.{field} differs")
    claim_raw = _stable_read(
        Path(claim["path"]),
        "release campaign claim",
        1_048_576,
        require_immutable=True,
    )
    authorization_raw = _stable_read(
        Path(claim["authorization_path"]),
        "release campaign authorization",
        1_048_576,
        require_immutable=True,
    )
    if (
        len(claim_raw) != claim["bytes"]
        or hashlib.sha256(claim_raw).hexdigest() != claim["sha256"]
        or hashlib.sha256(authorization_raw).hexdigest()
        != claim["authorization_sha256"]
    ):
        raise ValidationError("release campaign authorization bytes differ")

    binding = _decode_json(
        raws["active-manifest-binding.json"],
        "release campaign active binding",
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
        "release campaign active binding",
    )
    candidate = _exact_object(
        binding["candidate"],
        {"artifact", "source_path", "sha256", "bytes"},
        "release campaign binding candidate",
    )
    observations_reference = _exact_object(
        binding["observations"],
        {"artifact", "sha256", "bytes"},
        "release campaign binding observations",
    )
    binding_campaign = _exact_object(
        binding["campaign"],
        {"name", "run_id", "final_path"},
        "release campaign binding campaign",
    )
    candidate_raw = raws["candidate-served-model.json"]
    observations_raw = raws["active-manifest-observations.jsonl"]
    if (
        binding["schema_version"] != ACTIVE_BINDING_SCHEMA
        or binding["status"] != "complete"
        or binding["actual_active_path"] != FIXED_ACTIVE_MANIFEST_PATH
        or binding["expected_stages"] != list(REASONING_CAMPAIGN_STAGES)
        or binding["observation_count"] != len(REASONING_CAMPAIGN_STAGES)
        or binding["claim"] != claim
        or binding_campaign
        != {
            "name": "reasoning_release",
            "run_id": campaign["run_id"],
            "final_path": campaign["final_path"],
        }
        or candidate["artifact"] != "candidate-served-model.json"
        or candidate["sha256"] != hashlib.sha256(candidate_raw).hexdigest()
        or candidate["bytes"] != len(candidate_raw)
        or observations_reference
        != {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": hashlib.sha256(observations_raw).hexdigest(),
            "bytes": len(observations_raw),
        }
    ):
        raise ValidationError("release campaign active binding differs")

    observations = _exact_object(
        lineage["observations"],
        {"count", "stages"},
        "release campaign observation lineage",
    )
    lines = observations_raw.splitlines(keepends=True)
    if (
        observations["count"] != len(REASONING_CAMPAIGN_STAGES)
        or not isinstance(observations["stages"], list)
        or len(observations["stages"]) != len(REASONING_CAMPAIGN_STAGES)
        or len(lines) != len(REASONING_CAMPAIGN_STAGES)
        or any(not line.endswith(b"\n") for line in lines)
    ):
        raise ValidationError("release campaign observation count differs")
    for sequence, (stage, raw_line, expected_stage) in enumerate(
        zip(
            observations["stages"],
            lines,
            REASONING_CAMPAIGN_STAGES,
            strict=True,
        )
    ):
        stage = _exact_object(
            stage,
            {"sequence", "stage", "sha256"},
            f"release campaign observation lineage {sequence}",
        )
        _hash(stage["sha256"], f"release campaign observation lineage {sequence}.sha256")
        row = _decode_json(
            raw_line,
            f"release campaign observation {sequence}",
        )
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
            f"release campaign observation {sequence}",
        )
        candidate_row = _exact_object(
            row["candidate"],
            {"path", "sha256", "identity"},
            f"release campaign observation {sequence} candidate",
        )
        active_row = _exact_object(
            row["active"],
            {"path", "sha256", "identity"},
            f"release campaign observation {sequence} active",
        )
        if (
            stage
            != {
                "sequence": sequence,
                "stage": expected_stage,
                "sha256": hashlib.sha256(raw_line).hexdigest(),
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
            raise ValidationError(
                f"release campaign observation {sequence} differs"
            )
        for label, file_row in (("candidate", candidate_row), ("active", active_row)):
            identity = _exact_object(
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
                f"release campaign observation {sequence} {label} identity",
            )
            if (
                any(type(item) is not int or item < 0 for item in identity.values())
                or identity["bytes"] != candidate["bytes"]
            ):
                raise ValidationError(
                    f"release campaign observation {sequence} identity differs"
                )

    summary = _decode_json(raws["summary.json"], "release campaign summary")
    if (
        summary.get("schema_version") != REASONING_CAMPAIGN_SCHEMA_V2
        or summary.get("run_id") != campaign["run_id"]
        or summary.get("active_manifest_binding") != binding
        or summary.get("manifest_sha256") != candidate["sha256"]
        or summary.get("raw_bodies_stored") is not False
    ):
        raise ValidationError("release campaign summary differs")
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


def validate(path: Path) -> dict[str, Any]:
    document = _load(path)
    _scan_forbidden(document)
    expected = {
        "schema_version",
        "status",
        "production_activation_performed",
        "source_commit",
        "active_promotion_source_commit",
        "source_commit_aligned",
        "git_worktree_clean",
        "git_worktree_status_sha256",
        "identity",
        "cases",
        "lifecycle",
    }
    version = document.get("schema_version")
    if version == SCHEMA_VERSION_V2:
        expected.add("campaign_lineage")
    elif version != SCHEMA_VERSION_V1:
        raise ValidationError("release evidence schema is unsupported")
    if set(document) != expected:
        raise ValidationError("release evidence root fields differ")
    if document["status"] not in {"incomplete", "complete"}:
        raise ValidationError("release evidence status is invalid")
    if document["production_activation_performed"] is not False:
        raise ValidationError("release evidence claims activation")
    _commit(document["source_commit"], "source_commit")
    _commit(
        document["active_promotion_source_commit"],
        "active_promotion_source_commit",
    )
    if type(document["source_commit_aligned"]) is not bool:
        raise ValidationError("source alignment is invalid")
    computed_source_alignment = (
        document["source_commit"] == document["active_promotion_source_commit"]
    )
    if document["source_commit_aligned"] != computed_source_alignment:
        raise ValidationError("source alignment declaration differs from commit identity")
    if type(document["git_worktree_clean"]) is not bool:
        raise ValidationError("Git worktree clean declaration is invalid")
    _hash(document["git_worktree_status_sha256"], "git_worktree_status_sha256")
    _validate_identity(document["identity"])
    cases = document["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise ValidationError("release evidence cases are missing")
    modes: set[str] = set()
    ids: set[str] = set()
    cases_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        mode = _validate_case(case)
        if case["id"] in ids:
            raise ValidationError("release evidence case IDs are duplicated")
        ids.add(case["id"])
        cases_by_id[case["id"]] = case
        modes.add(mode)
    lifecycle = _validate_lifecycle(document["lifecycle"], cases_by_id)
    campaign_lineage = (
        _validate_campaign_lineage(
            document["campaign_lineage"],
            cases=cases,
            lifecycle=document["lifecycle"],
        )
        if version == SCHEMA_VERSION_V2
        else None
    )
    reasons: list[str] = []
    timing_fields = (
        "prefill_tokens_per_second",
        "first_reasoning_token_ms",
        "first_answer_token_ms",
        "reasoning_decode_tokens_per_second",
        "answer_decode_tokens_per_second",
        "decode_tokens_per_second",
        "latency_ms",
    )
    timing_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    quality_samples: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    resource_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in cases:
        for field in timing_fields:
            value = case["timing"][field]
            if value is not None:
                timing_samples[case["mode"]][field].append(float(value))
        quality_samples[case["mode"]]["total"] += 1
        if case["quality"]["correct"]:
            quality_samples[case["mode"]]["correct"] += 1
        for field, value in case["resource"].items():
            resource_samples[case["mode"]][field].append(float(value))
    if not computed_source_alignment:
        reasons.append("source commit is not aligned with the active promotion source")
    if document["git_worktree_clean"] is not True:
        reasons.append("Git worktree is not clean")
    missing_modes = sorted(REQUIRED_MODES - modes)
    if missing_modes:
        reasons.append("required benchmark modes are missing: " + ", ".join(missing_modes))
    if document["status"] != "complete":
        reasons.append("producer status is incomplete")
    if lifecycle["case_ids"] != ids:
        reasons.append("lifecycle evidence does not cover every release case")
    required_timing_fields = {
        "prefill_tokens_per_second",
        "first_answer_token_ms",
        "answer_decode_tokens_per_second",
        "decode_tokens_per_second",
        "latency_ms",
    }
    for case in cases:
        if case["quality"]["correct"] is not True:
            reasons.append(f"case quality is incorrect: {case['id']}")
        missing_timing = sorted(
            field
            for field in required_timing_fields
            if case["timing"][field] is None
        )
        if missing_timing:
            reasons.append(
                f"case timing is incomplete: {case['id']} ({', '.join(missing_timing)})"
            )
    return {
        "schema_version": (
            VALIDATOR_SCHEMA_VERSION_V2
            if version == SCHEMA_VERSION_V2
            else VALIDATOR_SCHEMA_VERSION
        ),
        "input_schema_version": version,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "case_count": len(cases),
        "lifecycle_event_count": lifecycle["event_count"],
        "git_worktree_clean": document["git_worktree_clean"],
        "observed_modes": sorted(modes),
        "timing_percentiles": {
            mode: {
                field: {
                    "count": len(values),
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "p99": _percentile(values, 0.99),
                }
                for field, values in sorted(fields.items())
            }
            for mode, fields in sorted(timing_samples.items())
        },
        "quality_summary": {
            mode: {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"],
            }
            for mode, values in sorted(quality_samples.items())
        },
        "resource_percentiles": {
            mode: {
                field: {
                    "count": len(values),
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "p99": _percentile(values, 0.99),
                    "maximum": max(values),
                }
                for field, values in sorted(fields.items())
            }
            for mode, fields in sorted(resource_samples.items())
        },
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
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.evidence)
    except Exception as error:
        print(f"Generic reasoning release validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["gate_eligible"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
