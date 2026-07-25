#!/usr/bin/env python3
"""Strict multi-step full-model gate for the private SQ8_0 WMMA prototype."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ck-result", required=True, type=Path)
    parser.add_argument("--handwritten-result", required=True, type=Path)
    parser.add_argument("--min-decode-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def one_request(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    requests = payload.get("requests")
    require(isinstance(requests, list) and len(requests) == 1, f"{path} must have one request")
    request = requests[0]
    require(isinstance(request, dict), f"{path} request must be an object")
    return request


def resolve(result_json: Path, value: Any) -> Path:
    require(isinstance(value, str), f"capture path in {result_json} must be a string")
    path = Path(value)
    return path if path.is_absolute() else result_json.parent / path


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    require(len(data) % 4 == 0, f"{path} is not whole f32le values")
    values = array.array("f")
    values.frombytes(data)
    if values.itemsize != 4:
        raise ValueError("platform float array does not have 32-bit items")
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        values.byteswap()
    return values.tolist()


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def compare_f32(ck_path: Path, handwritten_path: Path) -> dict[str, Any]:
    ck_values = read_f32(ck_path)
    handwritten_values = read_f32(handwritten_path)
    require(
        len(ck_values) == len(handwritten_values),
        f"capture length mismatch: {ck_path}={len(ck_values)} {handwritten_path}={len(handwritten_values)}",
    )
    mismatches = 0
    first_mismatch: int | None = None
    maximum_abs = 0.0
    maximum_rel = 0.0
    finite = True
    for index, (left, right) in enumerate(zip(ck_values, handwritten_values, strict=True)):
        finite = finite and math.isfinite(left) and math.isfinite(right)
        if f32_bits(left) != f32_bits(right):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = index
        difference = abs(left - right)
        maximum_abs = max(maximum_abs, difference)
        maximum_rel = max(maximum_rel, difference / max(abs(left), 1.0e-30))
    return {
        "elements": len(ck_values),
        "finite": finite,
        "bitwise_mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "max_abs": maximum_abs,
        "max_rel": maximum_rel,
        "ck_sha256": hashlib.sha256(ck_path.read_bytes()).hexdigest(),
        "handwritten_sha256": hashlib.sha256(handwritten_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    require(args.min_decode_steps >= 2, "--min-decode-steps must be at least two")
    ck = read_json(args.ck_result)
    handwritten = read_json(args.handwritten_result)
    require(
        ck.get("handwritten_wmma_projection_prototype") is False,
        "CK result must explicitly record handwritten_wmma_projection_prototype=false",
    )
    require(
        handwritten.get("handwritten_wmma_projection_prototype") is True,
        "candidate result must explicitly record handwritten_wmma_projection_prototype=true",
    )
    require(
        handwritten.get("schema_version") == "ullm.sq8.serving_handwritten_wmma_prototype.v1",
        "candidate result has the wrong handwritten-prototype schema",
    )
    ck_request = one_request(ck, args.ck_result)
    handwritten_request = one_request(handwritten, args.handwritten_result)
    require(
        ck_request.get("prompt_token_ids") == handwritten_request.get("prompt_token_ids"),
        "CK and candidate prompt token IDs differ",
    )
    require(
        ck_request.get("max_new_tokens") == handwritten_request.get("max_new_tokens"),
        "CK and candidate maximum generation lengths differ",
    )
    ck_generated = ck_request.get("generated_token_ids")
    handwritten_generated = handwritten_request.get("generated_token_ids")
    require(isinstance(ck_generated, list), "CK generated_token_ids must be a list")
    require(isinstance(handwritten_generated, list), "candidate generated_token_ids must be a list")
    ck_captures = ck_request.get("decode_oracle_captures")
    handwritten_captures = handwritten_request.get("decode_oracle_captures")
    require(isinstance(ck_captures, list), "CK result omitted decode oracle captures")
    require(isinstance(handwritten_captures, list), "candidate result omitted decode oracle captures")
    require(
        len(ck_captures) >= args.min_decode_steps and len(handwritten_captures) >= args.min_decode_steps,
        "multi-step gate requires the requested number of completed decode captures",
    )
    require(len(ck_captures) == len(handwritten_captures), "CK/candidate decode capture counts differ")

    steps: list[dict[str, Any]] = []
    pass_gate = ck_generated == handwritten_generated
    for index, (ck_capture, handwritten_capture) in enumerate(
        zip(ck_captures, handwritten_captures, strict=True), start=1
    ):
        require(isinstance(ck_capture, dict), f"CK capture {index} is not an object")
        require(isinstance(handwritten_capture, dict), f"candidate capture {index} is not an object")
        metadata_equal = all(
            ck_capture.get(field) == handwritten_capture.get(field)
            for field in ("generated_index", "cache_len", "position", "top1_token_id")
        )
        ck_top1_bits = f32_bits(float(ck_capture.get("top1_logit")))
        handwritten_top1_bits = f32_bits(float(handwritten_capture.get("top1_logit")))
        hidden = compare_f32(
            resolve(args.ck_result, ck_capture.get("final_hidden_file")),
            resolve(args.handwritten_result, handwritten_capture.get("final_hidden_file")),
        )
        logits = compare_f32(
            resolve(args.ck_result, ck_capture.get("logits_file")),
            resolve(args.handwritten_result, handwritten_capture.get("logits_file")),
        )
        step_pass = (
            metadata_equal
            and ck_top1_bits == handwritten_top1_bits
            and hidden["finite"]
            and logits["finite"]
            and hidden["bitwise_mismatches"] == 0
            and logits["bitwise_mismatches"] == 0
        )
        pass_gate = pass_gate and step_pass
        steps.append(
            {
                "decode_step": index,
                "metadata_equal": metadata_equal,
                "top1_logit_bitwise_equal": ck_top1_bits == handwritten_top1_bits,
                "hidden": hidden,
                "logits": logits,
                "passed": step_pass,
            }
        )

    result = {
        "schema_version": "ullm.sq8_0.handwritten_projection_multistep_gate.v1",
        "criterion": (
            "Candidate must execute as the actual full-model M=1 projection path for at least two "
            "feedback decode steps; generated IDs, top-1 logits, final hidden, and logits must be "
            "bitwise identical to the CK control at every captured step."
        ),
        "min_decode_steps": args.min_decode_steps,
        "ck_generated_token_ids": ck_generated,
        "handwritten_generated_token_ids": handwritten_generated,
        "generated_token_ids_equal": ck_generated == handwritten_generated,
        "steps": steps,
        "passed": pass_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0 if pass_gate else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"compare-sq8_0-handwritten-projection-gate: {error}")
