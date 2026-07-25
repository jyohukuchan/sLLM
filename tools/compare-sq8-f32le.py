#!/usr/bin/env python3
"""Compare two little-endian f32 vectors using the frozen SQ8 correctness gate.

The tool streams both files to avoid treating an oracle capture as a text
format.  It is intentionally general enough to compare the final-hidden and
logits captures made by ``sq8_ck_serving``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


SCHEMA = "ullm.sq8_0.f32le_differential.v0.1"
CHUNK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference: Path, candidate: Path, *, max_abs_gate: float, relative_l2_gate: float,
            cosine_gate: float) -> dict[str, object]:
    reference_size = reference.stat().st_size
    candidate_size = candidate.stat().st_size
    if reference_size == 0 or reference_size % 4 != 0:
        raise ValueError(f"reference is not a nonempty f32le vector: {reference}")
    if candidate_size != reference_size:
        raise ValueError(
            f"length mismatch: reference={reference_size} bytes candidate={candidate_size} bytes"
        )

    count = 0
    bit_mismatch_count = 0
    nonfinite_count = 0
    max_abs = 0.0
    max_abs_index = 0
    squared_error = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    with reference.open("rb") as left, candidate.open("rb") as right:
        while True:
            left_bytes = left.read(CHUNK_BYTES)
            right_bytes = right.read(CHUNK_BYTES)
            if not left_bytes:
                break
            if len(left_bytes) != len(right_bytes) or len(left_bytes) % 4 != 0:
                raise ValueError("inconsistent f32le reads")
            if left_bytes != right_bytes:
                for local_index, (expected_bits, actual_bits, expected, actual) in enumerate(
                    zip(
                        struct.iter_unpack("<I", left_bytes),
                        struct.iter_unpack("<I", right_bytes),
                        struct.iter_unpack("<f", left_bytes),
                        struct.iter_unpack("<f", right_bytes),
                    )
                ):
                    expected_value = expected[0]
                    actual_value = actual[0]
                    if expected_bits[0] != actual_bits[0]:
                        bit_mismatch_count += 1
                    if not math.isfinite(expected_value) or not math.isfinite(actual_value):
                        nonfinite_count += 1
                        continue
                    delta = actual_value - expected_value
                    absolute = abs(delta)
                    if absolute > max_abs:
                        max_abs = absolute
                        max_abs_index = count + local_index
                    squared_error += delta * delta
                    reference_squared += expected_value * expected_value
                    candidate_squared += actual_value * actual_value
                    dot += expected_value * actual_value
            else:
                for value in struct.iter_unpack("<f", left_bytes):
                    scalar = value[0]
                    if not math.isfinite(scalar):
                        nonfinite_count += 1
                        continue
                    reference_squared += scalar * scalar
                    candidate_squared += scalar * scalar
                    dot += scalar * scalar
            count += len(left_bytes) // 4

    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(reference_squared), 1.0e-30)
    cosine_denominator = math.sqrt(reference_squared) * math.sqrt(candidate_squared)
    cosine_similarity = dot / cosine_denominator if cosine_denominator else 1.0
    passed = (
        nonfinite_count == 0
        and math.isfinite(max_abs)
        and math.isfinite(relative_l2)
        and math.isfinite(cosine_similarity)
        and max_abs <= max_abs_gate
        and relative_l2 <= relative_l2_gate
        and cosine_similarity >= cosine_gate
    )
    return {
        "element_count": count,
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "bit_mismatch_count": bit_mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs": max_abs,
        "max_abs_index": max_abs_index,
        "relative_l2": relative_l2,
        "cosine_similarity": cosine_similarity,
        "thresholds": {
            "max_abs": max_abs_gate,
            "relative_l2": relative_l2_gate,
            "cosine_similarity": cosine_gate,
        },
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--max-abs", default=2.0e-5, type=float)
    parser.add_argument("--relative-l2", default=1.0e-5, type=float)
    parser.add_argument("--cosine", default=0.999999, type=float)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.result_json}")
    if args.max_abs < 0.0 or args.relative_l2 < 0.0 or not -1.0 <= args.cosine <= 1.0:
        raise SystemExit("invalid comparison threshold")
    result = {
        "schema_version": SCHEMA,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "metrics": compare(
            args.reference,
            args.candidate,
            max_abs_gate=args.max_abs,
            relative_l2_gate=args.relative_l2,
            cosine_gate=args.cosine,
        ),
    }
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["metrics"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
