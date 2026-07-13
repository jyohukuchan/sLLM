#!/usr/bin/env python3
"""Independently validate the hash-only generic reasoning Phase 0 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


TARGETS = (18, 1024, 2048, 3072)
SCHEMA_VERSION = "ullm.generic_reasoning_phase0_http_baseline.v1"
VALIDATOR_SCHEMA_VERSION = "ullm.generic_reasoning_phase0_http_baseline_validator.v1"
HASH_FIELDS = ("prompt_sha256", "request_body_sha256", "response_body_sha256")
FORBIDDEN_KEYS = {
    "prompt",
    "response",
    "request_body",
    "response_body",
    "authorization",
    "api_key",
    "api_key_value",
    "messages",
}


class ValidationError(ValueError):
    """Raised when a baseline record violates its published contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("baseline must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("baseline is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValidationError("baseline root must be an object")
    return value


def _scan_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValidationError(f"baseline contains forbidden field: {key}")
            _scan_keys(child)
    elif isinstance(value, list):
        for child in value:
            _scan_keys(child)


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValidationError(f"{label} is not a lowercase SHA-256")


def _nonnegative_integer(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValidationError(f"{label} is not a nonnegative integer")


def _validate_case(case: Any, target: int) -> None:
    if not isinstance(case, dict):
        raise ValidationError("baseline case is not an object")
    if case.get("id") != f"phase0-v1-target-{target}":
        raise ValidationError("baseline case ID differs from its target")
    if case.get("target_prompt_tokens") != target or case.get("prompt_tokens") != target:
        raise ValidationError("baseline prompt token count differs from its target")
    for field in ("prompt_sha256", "request_body_sha256"):
        _hash(case.get(field), f"case.{field}")
    nonstream = case.get("nonstream")
    if not isinstance(nonstream, dict):
        raise ValidationError("baseline non-stream metadata is missing")
    if nonstream.get("http_status") != 200:
        raise ValidationError("baseline non-stream request was not HTTP 200")
    if nonstream.get("prompt_tokens") != target:
        raise ValidationError("baseline non-stream usage prompt count differs")
    _nonnegative_integer(nonstream.get("completion_tokens"), "nonstream.completion_tokens")
    _nonnegative_integer(nonstream.get("total_tokens"), "nonstream.total_tokens")
    if nonstream["total_tokens"] != target + nonstream["completion_tokens"]:
        raise ValidationError("baseline non-stream usage total is inconsistent")
    _hash(nonstream.get("response_body_sha256"), "nonstream.response_body_sha256")
    _nonnegative_integer(nonstream.get("response_bytes"), "nonstream.response_bytes")


def _validate_stream(case: dict[str, Any]) -> None:
    stream = case.get("stream")
    if not isinstance(stream, dict):
        raise ValidationError("baseline stream metadata is missing")
    target = case["target_prompt_tokens"]
    if stream.get("http_status") != 200:
        raise ValidationError("baseline stream request was not HTTP 200")
    _hash(stream.get("request_body_sha256"), "stream.request_body_sha256")
    _hash(stream.get("response_body_sha256"), "stream.response_body_sha256")
    _nonnegative_integer(stream.get("chunks"), "stream.chunks")
    if stream.get("invalid_data_lines") != 0:
        raise ValidationError("baseline stream has invalid data lines")
    usage = stream.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != target:
        raise ValidationError("baseline stream usage prompt count differs")
    _nonnegative_integer(usage.get("completion_tokens"), "stream.completion_tokens")
    _nonnegative_integer(usage.get("total_tokens"), "stream.total_tokens")
    if usage["total_tokens"] != target + usage["completion_tokens"]:
        raise ValidationError("baseline stream usage total is inconsistent")
    sequence = stream.get("event_sequence")
    if (
        not isinstance(sequence, list)
        or not sequence
        or sequence[0] != "role"
        or sequence[-1] != "done"
        or "stop" not in sequence
        or "usage" not in sequence
        or sequence.index("stop") >= sequence.index("usage")
        or sequence.index("usage") >= len(sequence) - 1
    ):
        raise ValidationError("baseline SSE event sequence is incomplete")
    delta_keys = stream.get("delta_keys")
    if not isinstance(delta_keys, list) or not all(
        isinstance(keys, list) and all(isinstance(key, str) for key in keys)
        for keys in delta_keys
    ):
        raise ValidationError("baseline SSE delta keys are invalid")


def validate(path: Path) -> dict[str, Any]:
    document = _read_object(path)
    _scan_keys(document)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("baseline schema version differs")
    if document.get("production_activation_performed") is not False:
        raise ValidationError("baseline claims production activation")
    if document.get("raw_bodies_stored") is not False:
        raise ValidationError("baseline raw body policy is not hash-only")
    for field in ("source_commit", "active_promotion_source_commit", "endpoint", "image"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValidationError(f"baseline {field} is missing")
    _hash(document.get("active_manifest", {}).get("sha256"), "active_manifest.sha256")
    source_aligned = document.get("source_commit_aligned")
    if not isinstance(source_aligned, bool):
        raise ValidationError("baseline source alignment is not boolean")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != len(TARGETS):
        raise ValidationError("baseline case grid is incomplete")
    for case, target in zip(cases, TARGETS, strict=True):
        _validate_case(case, target)
    stream_cases = [case for case in cases if isinstance(case, dict) and "stream" in case]
    if len(stream_cases) != 1 or stream_cases[0]["target_prompt_tokens"] != TARGETS[0]:
        raise ValidationError("baseline stream case is missing or misplaced")
    _validate_stream(stream_cases[0])

    missing = document.get("missing")
    if not isinstance(missing, list) or not all(isinstance(item, str) and item for item in missing):
        raise ValidationError("baseline missing list is invalid")
    reasons: list[str] = []
    if not source_aligned:
        reasons.append("source commit is not aligned with the active promotion source")
    reasons.append("AQ4 generated token IDs are not present in this HTTP-only evidence")
    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "input_schema_version": SCHEMA_VERSION,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "source_commit_aligned": source_aligned,
        "case_count": len(cases),
        "reasons": reasons,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--require-complete", action="store_true", help="fail if the Phase 0 gate is incomplete"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.baseline)
    except Exception as error:
        print(f"Phase 0 baseline validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["gate_eligible"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
