#!/usr/bin/env python3
"""Validate bounded, hash-only generic reasoning release evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "ullm.generic_reasoning_release_evidence.v1"
VALIDATOR_SCHEMA_VERSION = "ullm.generic_reasoning_release_validator.v1"
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


class ValidationError(ValueError):
    """Raised when release evidence violates the published contract."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("release evidence must be a regular non-symlink file")
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_EVIDENCE_BYTES + 1)
        if len(raw) > MAX_EVIDENCE_BYTES:
            raise ValidationError("release evidence exceeds its size bound")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("release evidence is not valid JSON") from error
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
    }
    if set(document) != expected or document["schema_version"] != SCHEMA_VERSION:
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
    for case in cases:
        mode = _validate_case(case)
        if case["id"] in ids:
            raise ValidationError("release evidence case IDs are duplicated")
        ids.add(case["id"])
        modes.add(mode)
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
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "input_schema_version": SCHEMA_VERSION,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "case_count": len(cases),
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
