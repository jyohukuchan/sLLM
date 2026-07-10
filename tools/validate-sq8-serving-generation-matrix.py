#!/usr/bin/env python3
"""Validate SQ8 G=8/G=64 serving runs and compare frozen source token sequences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


RESULT_SCHEMA_VERSION = "ullm.sq8.serving_generation_matrix_validation.v1"
PROMPT_LENGTHS = (1, 8, 32, 128)
GENERATION_LENGTHS = (8, 64)
U32 = struct.Struct("<I")
REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-session-matrix.py"
ORACLE_VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-runtime-oracle.py"
DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT / "tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load validator module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load_module("sq8_serving_session_matrix", SESSION_VALIDATOR)
oracle = load_module("sq8_serving_runtime_oracle", ORACLE_VALIDATOR)
ValidationError = session.ValidationError


def fail(message: str) -> None:
    raise ValidationError(message)


def read_u32_tokens(path: Path, expected_tokens: int, label: str) -> list[int]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label} {path}: {error}")
    if len(payload) != expected_tokens * U32.size:
        fail(f"{label} has the wrong byte length")
    return [item[0] for item in struct.iter_unpack("<I", payload)]


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(U32.pack(token))
    return digest.hexdigest()


def common_prefix_length(actual: list[int], expected: list[int]) -> int:
    for index, (actual_token, expected_token) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if actual_token != expected_token:
            return index
    return len(actual)


def validate_generation_result(
    result: dict[str, Any],
    label: str,
    generation_tokens: int,
    source_oracle: dict[str, Any],
) -> list[dict[str, Any]]:
    requests = session.validate_common(result, label)
    if len(requests) != len(PROMPT_LENGTHS) or result.get("cancelled_request") is not None:
        fail(f"{label} must contain four completed requests and no cancellation")
    summaries = []
    for index, (request, prompt_tokens) in enumerate(
        zip(requests, PROMPT_LENGTHS, strict=True)
    ):
        prompt_id = f"raw-p{prompt_tokens:04d}"
        relative_path = f"prompts/{prompt_id}/greedy-g{generation_tokens}.u32le"
        reference_path, reference_sha256 = oracle.source_payload(
            source_oracle,
            relative_path,
            generation_tokens * U32.size,
        )
        expected = read_u32_tokens(
            reference_path, generation_tokens, f"{prompt_id} G={generation_tokens} source"
        )
        summary = session.validate_completed_request(
            request,
            prompt_tokens,
            f"serving-smoke-p{prompt_tokens:04}",
            f"{label}.requests[{index}]",
            max_new_tokens=generation_tokens,
            expected_generated=tuple(expected) if generation_tokens == 8 else None,
        )
        actual = summary["generated_token_ids"]
        if len(actual) != generation_tokens:
            fail(f"{label}.requests[{index}] did not complete G={generation_tokens}")
        if actual[0] != expected[0]:
            fail(f"{label}.requests[{index}] first token differs from the source oracle")
        common_prefix = common_prefix_length(actual, expected)
        summaries.append(
            {
                **summary,
                "generation_tokens": generation_tokens,
                "source_exact": actual == expected,
                "source_common_prefix_tokens": common_prefix,
                "first_source_difference_index": (
                    None if common_prefix == generation_tokens else common_prefix
                ),
                "actual_token_ids_sha256": token_sha256(actual),
                "source_token_ids_sha256": reference_sha256,
            }
        )
    return summaries


def validate_results(
    g8_path: Path, g64_path: Path, fixture_root: Path = DEFAULT_FIXTURE_ROOT
) -> dict[str, Any]:
    g8_file, g8_json = session.load_json(g8_path, "G=8 result")
    g64_file, g64_json = session.load_json(g64_path, "G=64 result")
    try:
        if os.path.samefile(g8_file, g64_file):
            fail("G=8 and G=64 evidence must be different files")
    except OSError as error:
        fail(f"failed to compare generation evidence identity: {error}")
    source_oracle = oracle.validate_source_oracle(fixture_root)
    g8 = validate_generation_result(g8_json, str(g8_file), 8, source_oracle)
    g64 = validate_generation_result(g64_json, str(g64_file), 64, source_oracle)
    if not all(item["source_exact"] for item in g8):
        fail("G=8 does not exactly reproduce all frozen source token sequences")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": True,
        "source_oracle": source_oracle["evidence"],
        "g8": g8,
        "g64": g64,
        "evidence": [
            {
                "generation_tokens": 8,
                "file": str(g8_file),
                "sha256": session.sha256_file(g8_file),
            },
            {
                "generation_tokens": 64,
                "file": str(g64_file),
                "sha256": session.sha256_file(g64_file),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("g8_result", type=Path)
    parser.add_argument("g64_result", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_results(args.g8_result, args.g64_result, args.fixture_root)
        if args.output is not None:
            session.write_json_create_new(args.output, result)
    except (ValidationError, oracle.ValidationError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    g64_prefixes = [item["source_common_prefix_tokens"] for item in result["g64"]]
    print(
        f"passed=true prompts={list(PROMPT_LENGTHS)} g8_exact=true "
        f"g64_completed=true g64_source_common_prefixes={g64_prefixes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
