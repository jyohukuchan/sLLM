#!/usr/bin/env python3
"""Validate hash-only OpenWebUI reasoning browser smoke evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "ullm.openwebui.reasoning_browser_smoke.v1"
VALIDATOR_SCHEMA_VERSION = "ullm.openwebui.reasoning_browser_smoke_validator.v1"
MAX_EVIDENCE_BYTES = 1 * 1024 * 1024
MAX_PROVIDER_REQUESTS = 4
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
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


class ValidationError(ValueError):
    """Raised when browser evidence violates the hash-only contract."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("browser evidence must be a regular non-symlink file")
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_EVIDENCE_BYTES + 1)
        if len(raw) > MAX_EVIDENCE_BYTES:
            raise ValidationError("browser evidence exceeds its size bound")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("browser evidence is not strict JSON") from error
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


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValidationError(f"{label} is invalid")


def _finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is invalid")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValidationError(f"{label} is invalid")


def _text_evidence(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"utf8_bytes", "sha256"}:
        raise ValidationError(f"{label} fields differ")
    _integer(value["utf8_bytes"], f"{label}.utf8_bytes", minimum=1, maximum=1_000_000)
    _hash(value["sha256"], f"{label}.sha256")


def _request(value: Any, index: int) -> None:
    if not isinstance(value, dict) or set(value) != {
        "sha256",
        "utf8_bytes",
        "has_reasoning_content_key",
        "assistant_has_reasoning_content",
    }:
        raise ValidationError(f"provider request {index} fields differ")
    _hash(value["sha256"], f"provider request {index}.sha256")
    _integer(value["utf8_bytes"], f"provider request {index}.utf8_bytes", minimum=2)
    if type(value["has_reasoning_content_key"]) is not bool or type(
        value["assistant_has_reasoning_content"]
    ) is not bool:
        raise ValidationError(f"provider request {index} flags are invalid")


def validate(path: Path) -> dict[str, Any]:
    document = _load(path)
    _scan_forbidden(document)
    expected = {
        "schema_version",
        "model_id_sha256",
        "first_answer",
        "second_answer",
        "reasoning_details_expanded",
        "provider_request_count",
        "provider_requests",
        "hidden_reasoning_reinserted",
        "page_error_count",
        "page_error_digests",
    }
    if set(document) != expected or document["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("browser evidence root fields differ")
    _hash(document["model_id_sha256"], "model_id_sha256")
    _text_evidence(document["first_answer"], "first_answer")
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
        _request(request, index)
    if document["hidden_reasoning_reinserted"] is not False:
        raise ValidationError("hidden reasoning was reinserted")
    _integer(document["page_error_count"], "page_error_count", maximum=0)
    page_errors = document["page_error_digests"]
    if not isinstance(page_errors, list) or page_errors:
        raise ValidationError("page error digests are not empty")
    reasons: list[str] = []
    if requests[-1]["assistant_has_reasoning_content"]:
        reasons.append("last provider request contains assistant reasoning_content")
    if not requests[-1]["has_reasoning_content_key"] and not any(
        request["has_reasoning_content_key"] for request in requests
    ):
        reasons.append("provider requests contain no reasoning field evidence")
    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "input_schema_version": SCHEMA_VERSION,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "provider_request_count": len(requests),
        "reasons": reasons,
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
