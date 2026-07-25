#!/usr/bin/env python3
"""Compare g0001 logical KV-cache prefix captures from two SQ8_0 routes.

The full-model tile gate captures final vectors only.  This companion keeps a
layer-by-layer state record so a decode feedback divergence can be located at
the first K/V prefix that differs.  It is diagnostic rather than an acceptance
gate: a non-bitwise match is written as evidence and does not itself make the
tool exit unsuccessfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "ullm.sq8_0.paged_decode_kv_prefix_diagnostic.v1"
CAPTURE_SCHEMA = "ullm.sq8_0.paged_decode_kv_prefix_capture.v1"
MAX_DIAGNOSTIC_GATE = 1.0e30
MIN_DIAGNOSTIC_COSINE = -1.0


class DiagnosticError(RuntimeError):
    """A malformed capture or comparison input."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"{label} is unreadable: {error}") from error
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def require_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a nonempty string")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_file(result_path: Path, recorded: Any, expected_hash: Any, label: str) -> Path:
    relative = Path(require_string(recorded, f"{label}.file"))
    path = relative if relative.is_absolute() else result_path.parent / relative
    try:
        path.resolve().relative_to(result_path.parent.resolve())
    except ValueError as error:
        raise DiagnosticError(f"{label}.file escapes result directory: {relative}") from error
    require(path.is_file(), f"{label}.file does not name a regular file: {path}")
    expected = require_string(expected_hash, f"{label}.sha256")
    actual = sha256_file(path)
    require(actual == expected, f"{label}.sha256 differs: expected={expected} actual={actual}")
    require(path.stat().st_size > 0 and path.stat().st_size % 4 == 0, f"{label}.file is not f32le")
    return path


def load_capture(result_path: Path) -> dict[str, Any]:
    result = load_json(result_path, str(result_path))
    requests = result.get("requests")
    require(isinstance(requests, list) and len(requests) == 1, "diagnostic result must contain one request")
    request = requests[0]
    require(isinstance(request, dict), "diagnostic request must be an object")
    capture = request.get("kv_cache_prefix_capture")
    require(isinstance(capture, dict), "diagnostic result omits kv_cache_prefix_capture")
    require(capture.get("schema_version") == CAPTURE_SCHEMA, "KV prefix capture schema differs")
    generated_index = require_int(capture.get("generated_index"), "kv.generated_index")
    cache_len = require_int(capture.get("cache_len"), "kv.cache_len")
    layer_count = require_int(capture.get("layer_count"), "kv.layer_count")
    require(generated_index == 1, "KV prefix diagnostic must capture g0001")
    require(cache_len == len(request.get("prompt_token_ids", [])) + 1, "KV cache geometry differs")
    layers = capture.get("layers")
    require(isinstance(layers, list) and len(layers) == layer_count, "KV layer count differs")

    loaded_layers: list[dict[str, Any]] = []
    for expected_index, layer in enumerate(layers):
        require(isinstance(layer, dict), f"kv.layers[{expected_index}] must be an object")
        layer_index = require_int(layer.get("layer_index"), f"kv.layers[{expected_index}].layer_index")
        require(layer_index == expected_index, f"kv.layers[{expected_index}].layer_index differs")
        k_elements = require_int(layer.get("k_elements"), f"kv.layers[{expected_index}].k_elements")
        v_elements = require_int(layer.get("v_elements"), f"kv.layers[{expected_index}].v_elements")
        k_path = capture_file(
            result_path,
            layer.get("k_file"),
            layer.get("k_f32_le_sha256"),
            f"kv.layers[{expected_index}].k",
        )
        v_path = capture_file(
            result_path,
            layer.get("v_file"),
            layer.get("v_f32_le_sha256"),
            f"kv.layers[{expected_index}].v",
        )
        require(k_path.stat().st_size == k_elements * 4, f"KV K element count differs at layer {layer_index}")
        require(v_path.stat().st_size == v_elements * 4, f"KV V element count differs at layer {layer_index}")
        loaded_layers.append(
            {
                "layer_index": layer_index,
                "k": k_path,
                "v": v_path,
                "k_elements": k_elements,
                "v_elements": v_elements,
            }
        )
    return {
        "request_id": require_string(request.get("request_id"), "request_id"),
        "cache_len": cache_len,
        "layers": loaded_layers,
    }


def load_comparator() -> Any:
    path = Path(__file__).with_name("compare-sq8-f32le.py")
    specification = importlib.util.spec_from_file_location("sq8_f32le_comparator", path)
    if specification is None or specification.loader is None:
        raise DiagnosticError(f"cannot load comparator {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "element_count",
            "reference_sha256",
            "candidate_sha256",
            "bit_mismatch_count",
            "nonfinite_count",
            "max_abs",
            "max_abs_index",
            "relative_l2",
            "cosine_similarity",
        )
    }


def compare(
    direct_result: Path,
    candidate_result: Path,
    comparator: Any,
) -> dict[str, Any]:
    direct = load_capture(direct_result)
    candidate = load_capture(candidate_result)
    require(direct["request_id"] == candidate["request_id"], "diagnostic request IDs differ")
    require(direct["cache_len"] == candidate["cache_len"], "diagnostic cache lengths differ")
    require(len(direct["layers"]) == len(candidate["layers"]), "diagnostic layer counts differ")

    rows: list[dict[str, Any]] = []
    all_finite = True
    all_bitwise_equal = True
    first_difference: dict[str, Any] | None = None
    worst_max_abs = 0.0
    worst_relative_l2 = 0.0
    minimum_cosine = 1.0
    for reference_layer, candidate_layer in zip(direct["layers"], candidate["layers"], strict=True):
        layer_index = reference_layer["layer_index"]
        require(layer_index == candidate_layer["layer_index"], "diagnostic layer ordering differs")
        layer_result: dict[str, Any] = {"layer_index": layer_index}
        for component in ("k", "v"):
            metrics = comparator.compare(
                reference_layer[component],
                candidate_layer[component],
                max_abs_gate=MAX_DIAGNOSTIC_GATE,
                relative_l2_gate=MAX_DIAGNOSTIC_GATE,
                cosine_gate=MIN_DIAGNOSTIC_COSINE,
            )
            compact = compact_metrics(metrics)
            layer_result[component] = compact
            finite = compact["nonfinite_count"] == 0
            equal = compact["bit_mismatch_count"] == 0
            all_finite = all_finite and finite
            all_bitwise_equal = all_bitwise_equal and equal
            worst_max_abs = max(worst_max_abs, float(compact["max_abs"]))
            worst_relative_l2 = max(worst_relative_l2, float(compact["relative_l2"]))
            minimum_cosine = min(minimum_cosine, float(compact["cosine_similarity"]))
            if first_difference is None and not equal:
                first_difference = {
                    "layer_index": layer_index,
                    "component": component,
                    "max_abs": compact["max_abs"],
                    "max_abs_index": compact["max_abs_index"],
                    "bit_mismatch_count": compact["bit_mismatch_count"],
                }
        rows.append(layer_result)
    require(math.isfinite(worst_max_abs), "diagnostic maximum absolute difference is non-finite")
    require(math.isfinite(worst_relative_l2), "diagnostic relative L2 is non-finite")
    require(math.isfinite(minimum_cosine), "diagnostic cosine is non-finite")
    return {
        "schema_version": SCHEMA,
        "reference_result": str(direct_result),
        "candidate_result": str(candidate_result),
        "request_id": direct["request_id"],
        "generated_index": 1,
        "cache_len": direct["cache_len"],
        "layer_count": len(rows),
        "all_values_finite": all_finite,
        "all_bitwise_equal": all_bitwise_equal,
        "first_difference": first_difference,
        "worst_max_abs": worst_max_abs,
        "worst_relative_l2": worst_relative_l2,
        "minimum_cosine_similarity": minimum_cosine,
        "layers": rows,
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-result", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing diagnostic: {args.output}")
    try:
        summary = compare(
            args.direct_result.resolve(),
            args.candidate_result.resolve(),
            load_comparator(),
        )
        write_json_new(args.output, summary)
    except DiagnosticError as error:
        raise SystemExit(f"KV prefix diagnostic failed: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
