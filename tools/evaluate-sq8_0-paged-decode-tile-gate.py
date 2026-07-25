#!/usr/bin/env python3
"""Evaluate the frozen full-model SQ8_0 paged-decode tile numerical gate.

The direct legacy route is the only reference.  Every requested real-prompt
decode capture must be present before a candidate route can pass: greedy token
IDs must match exactly and both final-hidden and logits vectors are compared
with the criteria saved before the GPU window started.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
from typing import Any


SCHEMA = "ullm.sq8_0.paged_decode_tile_gate_summary.v1"
EXPECTED_ARCH = "gfx1201"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
# These are the full-model output shapes, not merely a nonempty-vector
# sentinel.  Keeping them here makes a truncated oracle capture fail closed.
FINAL_HIDDEN_ELEMENTS = 5_120
LOGITS_ELEMENTS = 151_936
# Do not let a pre-window criteria file silently relax the predeclared gate.
MAX_ALLOWED_MAX_ABS = 2.0e-5
MAX_ALLOWED_RELATIVE_L2 = 1.0e-5
MIN_ALLOWED_COSINE = 0.999999


class GateError(RuntimeError):
    """A malformed or failed gate input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{label} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GateError(f"{label} must be an integer")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a nonempty string")
    return value


def relative_capture_path(result_path: Path, recorded: Any, label: str) -> Path:
    value = Path(require_string(recorded, label))
    path = value if value.is_absolute() else result_path.parent / value
    try:
        path.resolve().relative_to(result_path.parent.resolve())
    except ValueError as error:
        raise GateError(f"{label} escapes result directory: {value}") from error
    if not path.is_file():
        raise GateError(f"{label} does not name a regular file: {path}")
    return path


def greedy_top1_f32le(path: Path) -> int:
    """Recompute the deterministic greedy token from a captured logits vector."""
    best_index: int | None = None
    best_value = 0.0
    index = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            require(len(block) % 4 == 0, f"logits capture has a partial f32: {path}")
            for (value,) in struct.iter_unpack("<f", block):
                require(math.isfinite(value), f"logits capture has a non-finite value at {index}")
                if best_index is None or value > best_value:
                    best_index = index
                    best_value = value
                index += 1
    require(best_index is not None, f"logits capture is empty: {path}")
    return best_index


def load_comparator() -> Any:
    path = Path(__file__).with_name("compare-sq8-f32le.py")
    specification = importlib.util.spec_from_file_location("sq8_f32le_comparator", path)
    if specification is None or specification.loader is None:
        raise GateError(f"cannot load vector comparator {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def capture_paths(
    capture: dict[str, Any],
    result_path: Path,
    expected_generated_index: int,
    expected_cache_len: int,
    label: str,
) -> tuple[Path, Path]:
    generated_index = require_int(capture.get("generated_index"), f"{label}.generated_index")
    cache_len = require_int(capture.get("cache_len"), f"{label}.cache_len")
    position = require_int(capture.get("position"), f"{label}.position")
    top1_token_id = require_int(capture.get("top1_token_id"), f"{label}.top1_token_id")
    require(
        generated_index == expected_generated_index,
        f"{label}.generated_index={generated_index}, expected {expected_generated_index}",
    )
    require(
        cache_len == expected_cache_len,
        f"{label}.cache_len={cache_len}, expected {expected_cache_len}",
    )
    require(
        position + 1 == cache_len,
        f"{label}.position/cache_len are inconsistent",
    )
    hidden = relative_capture_path(
        result_path, capture.get("final_hidden_file"), f"{label}.final_hidden_file"
    )
    logits = relative_capture_path(result_path, capture.get("logits_file"), f"{label}.logits_file")
    for path, hash_key, vector_label, expected_elements in (
        (hidden, "final_hidden_f32_le_sha256", "final_hidden", FINAL_HIDDEN_ELEMENTS),
        (logits, "logits_f32_le_sha256", "logits", LOGITS_ELEMENTS),
    ):
        require(
            path.stat().st_size == expected_elements * 4,
            f"{label}.{vector_label} element count differs: "
            f"expected {expected_elements}, got {path.stat().st_size // 4}",
        )
        expected_hash = require_string(capture.get(hash_key), f"{label}.{hash_key}")
        actual_hash = sha256_file(path)
        require(
            actual_hash == expected_hash,
            f"{label}.{vector_label} SHA-256 differs: expected={expected_hash} actual={actual_hash}",
        )
    require(
        greedy_top1_f32le(logits) == top1_token_id,
        f"{label}.top1_token_id does not match the captured logits argmax",
    )
    # The evaluator checks exact token agreement after direct and candidate
    # cases are paired.  The per-capture top-1 must already agree with the
    # emitted greedy token for this step.
    capture["_validated_top1_token_id"] = top1_token_id
    return hidden, logits


def validate_request(
    request: dict[str, Any],
    expected: dict[str, Any],
    result_path: Path,
    generation: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    request_id = require_string(request.get("request_id"), f"{label}.request_id")
    require(request_id == expected["request_id"], f"{label}.request_id differs")
    prompt_tokens = request.get("prompt_token_ids")
    require(isinstance(prompt_tokens, list), f"{label}.prompt_token_ids must be a list")
    expected_prompt_tokens = require_int(expected.get("prompt_tokens"), f"{label}.expected_prompt_tokens")
    require(len(prompt_tokens) == expected_prompt_tokens, f"{label}.prompt token count differs")
    max_new_tokens = require_int(request.get("max_new_tokens"), f"{label}.max_new_tokens")
    require(
        max_new_tokens == require_int(generation.get("max_new_tokens"), "generation.max_new_tokens"),
        f"{label}.max_new_tokens differs from frozen criteria",
    )
    generated_tokens = request.get("generated_token_ids")
    require(isinstance(generated_tokens, list), f"{label}.generated_token_ids must be a list")
    require(
        len(generated_tokens) == max_new_tokens,
        f"{label} stopped before the required {max_new_tokens} generated tokens",
    )
    require(
        all(isinstance(token, int) and not isinstance(token, bool) for token in generated_tokens),
        f"{label}.generated_token_ids is not integer-only",
    )
    generated_steps = request.get("generated_steps")
    require(isinstance(generated_steps, list), f"{label}.generated_steps must be present")
    require(len(generated_steps) == max_new_tokens, f"{label}.generated_steps length differs")

    expected_cache_lengths = expected.get("decode_cache_lengths")
    require(
        isinstance(expected_cache_lengths, list) and expected_cache_lengths,
        f"{label}.decode_cache_lengths criteria is invalid",
    )
    captures = request.get("decode_oracle_captures")
    require(isinstance(captures, list), f"{label}.decode_oracle_captures must be present")
    required_indices = generation.get("captured_decode_indices")
    require(isinstance(required_indices, list), "generation.captured_decode_indices must be a list")
    require(len(captures) == len(required_indices), f"{label}.decode capture count differs")
    require(
        len(expected_cache_lengths) == len(required_indices),
        f"{label}.frozen cache geometry count differs",
    )

    capture_rows: list[dict[str, Any]] = []
    for offset, (generated_index, cache_len) in enumerate(
        zip(required_indices, expected_cache_lengths, strict=True)
    ):
        generated_index = require_int(generated_index, "generation.captured_decode_indices entry")
        cache_len = require_int(cache_len, f"{label}.decode_cache_lengths entry")
        require(
            cache_len == expected_prompt_tokens + generated_index,
            f"{label}.frozen cache geometry is inconsistent",
        )
        capture = captures[offset]
        require(isinstance(capture, dict), f"{label}.decode_oracle_captures[{offset}] must be an object")
        hidden, logits = capture_paths(
            capture,
            result_path,
            generated_index,
            cache_len,
            f"{label}.decode_oracle_captures[{offset}]",
        )
        require(
            generated_steps[generated_index].get("cache_len") == cache_len,
            f"{label}.generated_steps[{generated_index}] cache length differs",
        )
        require(
            generated_steps[generated_index].get("token_id") == generated_tokens[generated_index],
            f"{label}.generated_steps[{generated_index}] token differs",
        )
        require(
            capture["_validated_top1_token_id"] == generated_tokens[generated_index],
            f"{label}.decode capture top-1 differs from emitted token",
        )
        capture_rows.append(
            {
                "generated_index": generated_index,
                "cache_len": cache_len,
                "hidden": hidden,
                "logits": logits,
            }
        )
    return {
        "request_id": request_id,
        "prompt_token_ids": prompt_tokens,
        "generated_token_ids": generated_tokens,
        "captures": capture_rows,
    }


def load_route(
    root: Path,
    route: str,
    criteria: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    generation = criteria["generation"]
    loaded: dict[str, list[dict[str, Any]]] = {}
    for group in criteria["case_groups"]:
        group_name = require_string(group.get("name"), "case_groups.name")
        require(SAFE_COMPONENT.fullmatch(group_name) is not None, "case group name is unsafe")
        result_path = root / "cases" / route / group_name / "result.json"
        result = load_json(result_path, f"{route}/{group_name} result")
        require(result.get("passed") is True, f"{route}/{group_name} runner did not pass")
        require(
            result.get("prefill_mode") == "m128-chunk128"
            and result.get("prefill_chunk_tokens") == 128,
            f"{route}/{group_name} did not use the frozen M=128 prefill mode",
        )
        require(
            result.get("test_only_ignore_eos") is None,
            f"{route}/{group_name} used test-only EOS suppression",
        )
        require(
            result.get("cancelled_request") is None,
            f"{route}/{group_name} includes a cancelled request",
        )
        device = result.get("device")
        require(isinstance(device, dict), f"{route}/{group_name}.device is absent")
        require(
            device.get("gcn_arch_name") == EXPECTED_ARCH,
            f"{route}/{group_name} is not an {EXPECTED_ARCH} result",
        )
        expected_tile: int | None = None if route == criteria["reference_route"] else int(route.removeprefix("tile"))
        require(
            result.get("paged_decode_split_source_tile") == expected_tile,
            f"{route}/{group_name} dispatch selection differs",
        )
        requests = result.get("requests")
        expected_requests = group.get("requests")
        require(isinstance(requests, list), f"{route}/{group_name}.requests must be a list")
        require(isinstance(expected_requests, list), "case group requests must be a list")
        require(
            len(requests) == len(expected_requests),
            f"{route}/{group_name} request count differs",
        )
        loaded[group_name] = [
            validate_request(request, expected, result_path, generation, f"{route}/{group_name}/{index}")
            for index, (request, expected) in enumerate(zip(requests, expected_requests, strict=True))
            if isinstance(request, dict) and isinstance(expected, dict)
        ]
        require(
            len(loaded[group_name]) == len(expected_requests),
            f"{route}/{group_name} request shape is invalid",
        )
    return loaded


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def compare_route(
    root: Path,
    candidate_route: str,
    direct: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
    criteria: dict[str, Any],
    comparator: Any,
) -> dict[str, Any]:
    thresholds = criteria["thresholds"]
    token_exact = True
    comparisons: list[dict[str, Any]] = []
    for group in criteria["case_groups"]:
        group_name = group["name"]
        direct_requests = direct[group_name]
        candidate_requests = candidate[group_name]
        for direct_request, candidate_request in zip(direct_requests, candidate_requests, strict=True):
            label = f"{candidate_route}/{group_name}/{direct_request['request_id']}"
            same_prompt = direct_request["prompt_token_ids"] == candidate_request["prompt_token_ids"]
            same_tokens = direct_request["generated_token_ids"] == candidate_request["generated_token_ids"]
            token_exact = token_exact and same_prompt and same_tokens
            for reference_capture, candidate_capture in zip(
                direct_request["captures"], candidate_request["captures"], strict=True
            ):
                require(
                    reference_capture["generated_index"] == candidate_capture["generated_index"]
                    and reference_capture["cache_len"] == candidate_capture["cache_len"],
                    f"{label} decode geometry differs",
                )
                for vector_name in ("hidden", "logits"):
                    metrics = comparator.compare(
                        reference_capture[vector_name],
                        candidate_capture[vector_name],
                        max_abs_gate=float(thresholds["max_abs"]),
                        relative_l2_gate=float(thresholds["relative_l2"]),
                        cosine_gate=float(thresholds["cosine_similarity"]),
                    )
                    record = {
                        "schema_version": "ullm.sq8_0.paged_decode_tile_vector_gate.v1",
                        "route": candidate_route,
                        "group": group_name,
                        "request_id": direct_request["request_id"],
                        "generated_index": reference_capture["generated_index"],
                        "cache_len": reference_capture["cache_len"],
                        "vector": vector_name,
                        "reference": str(reference_capture[vector_name]),
                        "candidate": str(candidate_capture[vector_name]),
                        "metrics": metrics,
                    }
                    component = direct_request["request_id"]
                    require(SAFE_COMPONENT.fullmatch(component) is not None, "request id is unsafe")
                    output = (
                        root
                        / "comparisons"
                        / candidate_route
                        / group_name
                        / component
                        / f"g{reference_capture['generated_index']:04}-{vector_name}.json"
                    )
                    write_json_new(output, record)
                    comparisons.append(
                        {
                            "path": str(output.relative_to(root)),
                            "passed": metrics["passed"],
                            "max_abs": metrics["max_abs"],
                            "relative_l2": metrics["relative_l2"],
                            "cosine_similarity": metrics["cosine_similarity"],
                        }
                    )
    all_vectors_passed = all(row["passed"] for row in comparisons)
    return {
        "token_exact_match": token_exact,
        "vector_comparison_count": len(comparisons),
        "vector_comparisons": comparisons,
        "passed": token_exact and all_vectors_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    summary_path = root / "summary.json"
    if summary_path.exists():
        raise SystemExit(f"refusing to overwrite existing summary: {summary_path}")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "result_dir": str(root),
        "passed": False,
        "failures": [],
        "routes": {},
    }
    try:
        criteria_path = root / "gate-criteria.json"
        criteria = load_json(criteria_path, "frozen gate criteria")
        require(
            criteria.get("schema_version") == "ullm.sq8_0.paged_decode_tile_gate_criteria.v1",
            "gate criteria schema differs",
        )
        require(criteria.get("frozen_before_measurement") is True, "gate criteria is not frozen")
        require(criteria.get("reference_route") == "direct", "reference route must be direct")
        require(
            criteria.get("candidate_routes") == ["tile128", "tile256"],
            "candidate route set differs",
        )
        require(isinstance(criteria.get("generation"), dict), "generation criteria is absent")
        require(isinstance(criteria.get("thresholds"), dict), "threshold criteria is absent")
        require(isinstance(criteria.get("case_groups"), list), "case group criteria is absent")
        require(criteria["case_groups"], "case group criteria is empty")
        for key in ("max_abs", "relative_l2", "cosine_similarity"):
            value = criteria["thresholds"].get(key)
            require(isinstance(value, (int, float)) and not isinstance(value, bool), f"threshold {key} is invalid")
        thresholds = criteria["thresholds"]
        require(
            thresholds.get("all_values_must_be_finite") is True,
            "criteria must require finite vectors",
        )
        require(
            thresholds.get("token_exact_match") is True,
            "criteria must require exact greedy token agreement",
        )
        require(
            0.0 <= float(thresholds["max_abs"]) <= MAX_ALLOWED_MAX_ABS,
            "criteria max_abs is weaker than the frozen numerical gate",
        )
        require(
            0.0 <= float(thresholds["relative_l2"]) <= MAX_ALLOWED_RELATIVE_L2,
            "criteria relative_l2 is weaker than the frozen numerical gate",
        )
        require(
            MIN_ALLOWED_COSINE <= float(thresholds["cosine_similarity"]) <= 1.0,
            "criteria cosine_similarity is weaker than the frozen numerical gate",
        )
        summary["criteria_sha256"] = sha256_file(criteria_path)
        summary["criteria"] = criteria

        direct = load_route(root, criteria["reference_route"], criteria)
        comparator = load_comparator()
        for route in criteria["candidate_routes"]:
            candidate = load_route(root, route, criteria)
            summary["routes"][route] = compare_route(
                root, route, direct, candidate, criteria, comparator
            )
        summary["passed"] = all(route["passed"] for route in summary["routes"].values())
    except GateError as error:
        summary["failures"].append(str(error))
    except Exception as error:  # Keep a machine-readable failure record for a failed gate.
        summary["failures"].append(f"unexpected evaluator error: {type(error).__name__}: {error}")

    write_json_new(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
