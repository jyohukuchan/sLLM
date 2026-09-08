#!/usr/bin/env python3
"""Independent NVFP4 prefill arithmetic oracle for Phase 82 ID62.

This is deliberately a host-only model.  It does not call the runtime or a
GPU.  The two candidate reductions are reconstructed from the source-level
contracts of ID59 and ID62, and both are compared with an exact dyadic
reference.  It is intended to expose the error shape of a candidate before a
GPU run; it cannot establish dispatch, compiler, or model-logit behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Iterable


E2M1_VALUES = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
               Fraction(2), Fraction(3), Fraction(4), Fraction(6))
TENSOR_SCALE = Fraction(3, 4)


def f32(value: float) -> float:
    """Round one operation to IEEE binary32, matching a float register."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def e2m1(code: int) -> Fraction:
    value = E2M1_VALUES[code & 7]
    return -value if code & 8 else value


def e4m3fn(code: int) -> Fraction:
    """Decode the finite OCP E4M3FN values used by block16 scales."""

    negative = bool(code & 0x80)
    magnitude = code & 0x7F
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    if exponent == 0:
        value = Fraction(mantissa, 2**9)
    elif magnitude == 0x7F:
        raise ValueError(f"NaN E4M3FN code 0x{code:02x}")
    else:
        value = (Fraction(8 + mantissa, 8) *
                 (Fraction(2) ** (exponent - 7)))
    return -value if negative else value


def activation_code(row: int, inner: int) -> int:
    return (row * 3 + inner * 5 + 1) & 15


def weight_code(column: int, inner: int) -> int:
    return (column * 3 + inner * 7 + 5) & 15


def weight_scale_code(column: int, block: int) -> int:
    return 0x30 + ((column + block) % 4) * 8


def activation_scale_code(row: int, block: int) -> int:
    # The Phase79 runner uses 0x38 for its compact encoded fixture.  This
    # independent oracle also varies valid E4M3 block scales to exercise the
    # rounding boundary that a uniform unit scale cannot expose.
    return 0x28 + ((row * 3 + block * 5) % 9) * 8


def ref_value(row: int, column: int, k: int) -> float:
    """Exact real-valued encoded dot product, rounded only on return."""

    total = Fraction(0)
    for inner in range(k):
        activation_scale = e4m3fn(activation_scale_code(row, inner // 16))
        total += (e2m1(activation_code(row, inner)) * activation_scale *
                  e2m1(weight_code(column, inner)) *
                  e4m3fn(weight_scale_code(column, inner // 16)))
    return float(total * TENSOR_SCALE)


def id59_value(row: int, column: int, k: int) -> float:
    """Model ID59's 8-wave K256 tile and final shuffle tree."""

    if k % 16:
        raise ValueError("ID59 oracle requires K divisible by block16")
    partials = [f32(0.0) for _ in range(8 * 32)]
    for base in range(0, k, 256):
        valid = min(k - base, 256)
        # One wave lane owns each offset modulo 32.  A slot is accumulated
        # across K tiles before the source's wave shuffle reduction.
        for slot in range(8):
            for lane in range(32):
                inner = base + lane + slot * 32
                if inner >= base + valid:
                    continue
                a = f32(float(e2m1(activation_code(row, inner))))
                activation_scale = f32(float(e4m3fn(
                    activation_scale_code(row, inner // 16))))
                ws = f32(float(e4m3fn(weight_scale_code(column, inner // 16))))
                w = f32(float(e2m1(weight_code(column, inner))))
                weighted = f32(w * ws)
                term = f32(f32(a * activation_scale) * weighted)
                index = slot * 32 + lane
                partials[index] = f32(partials[index] + term)
    slot_sums: list[float] = []
    for slot in range(8):
        values = partials[slot * 32:(slot + 1) * 32]
        # __shfl_down reductions are simultaneous.  Only lanes below offset
        # change at each step; lane zero is the final wave sum.
        for offset in (16, 8, 4, 2, 1):
            old = values[:]
            for lane in range(offset):
                values[lane] = f32(old[lane] + old[lane + offset])
        slot_sums.append(values[0])
    sum0 = f32(slot_sums[0] + slot_sums[4])
    sum1 = f32(slot_sums[1] + slot_sums[5])
    sum2 = f32(slot_sums[2] + slot_sums[6])
    sum3 = f32(slot_sums[3] + slot_sums[7])
    accumulator = f32(f32(sum0 + sum2) + f32(sum1 + sum3))
    return f32(accumulator * f32(float(TENSOR_SCALE)))


def id62_value(row: int, column: int, k: int) -> float:
    """Model ID62's int8 dot4 block16 conversion and sequential accumulation."""

    if k % 16:
        raise ValueError("ID62 oracle requires K divisible by block16")
    accumulator = f32(0.0)
    for block_start in range(0, k, 16):
        block_sum = 0
        for inner in range(block_start, block_start + 16):
            # The source stores E2M1*2 as signed bytes.  DP4A is exact for
            # this range, and the 0.25 factor restores the original values.
            block_sum += (int(e2m1(activation_code(row, inner)) * 2) *
                          int(e2m1(weight_code(column, inner)) * 2))
        activation_scale = f32(float(e4m3fn(
            activation_scale_code(row, block_start // 16))) * 0.25)
        weight_scale = f32(float(e4m3fn(weight_scale_code(column,
                                                         block_start // 16))))
        term = f32(f32(float(block_sum) * activation_scale) * weight_scale)
        accumulator = f32(accumulator + term)
    return f32(accumulator * f32(float(TENSOR_SCALE)))


def sample_points(m: int, n: int, full: bool) -> list[tuple[int, int]]:
    if full:
        return [(row, column) for row in range(m) for column in range(n)]
    rows = sorted({0, min(1, m - 1), m // 2, m - 1})
    columns = sorted({0, min(1, n - 1), 31, 32, 63, 64, n - 1})
    return [(row, column) for row in rows for column in columns if column < n]


def metrics(values: Iterable[tuple[float, float]]) -> dict[str, object]:
    pairs = list(values)
    finite = all(math.isfinite(value) and math.isfinite(reference)
                 for value, reference in pairs)
    if not pairs:
        return {"count": 0, "finite": False}
    if not finite:
        return {"count": len(pairs), "finite": False}
    errors = [abs(value - reference) for value, reference in pairs]
    relative = [error / max(1.0, abs(reference))
                for error, (_, reference) in zip(errors, pairs)]
    return {
        "count": len(pairs),
        "finite": True,
        "max_abs": max(errors),
        "max_rel_denominator_max1": max(relative),
        "rms": math.sqrt(math.fsum(error * error for error in errors) /
                          len(errors)),
    }


def compare_shape(name: str, m: int, k: int, n: int, full: bool) -> dict:
    points = sample_points(m, n, full)
    rows: list[tuple[float, float, float]] = []
    for row, column in points:
        reference = ref_value(row, column, k)
        rows.append((id59_value(row, column, k),
                     id62_value(row, column, k), reference))
    baseline = [(left, reference) for left, _, reference in rows]
    candidate = [(right, reference) for _, right, reference in rows]
    pairwise = [(right, left) for left, right, _ in rows]
    return {
        "name": name,
        "shape": {"m": m, "k": k, "n": n},
        "sampling": "all_outputs" if full else "tile_boundary_points",
        "sample_points": len(points),
        "baseline_id59_vs_exact": metrics(baseline),
        "candidate_id62_vs_exact": metrics(candidate),
        "id62_minus_id59": metrics(pairwise),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON report path under .local-artifacts/phase82")
    args = parser.parse_args()
    shapes = [
        # The six Phase79 boundary shapes are retained.  Small cases are
        # exhaustive so N and M tail behaviour is visible in every element.
        ("boundary-m63-k48-n37", 63, 48, 37, True),
        ("boundary-m64-k48-n37", 64, 48, 37, True),
        ("boundary-m65-k48-n37", 65, 48, 37, True),
        ("boundary-m63-k3840-n15360", 63, 3840, 15360, False),
        ("boundary-m64-k3840-n15360", 64, 3840, 15360, False),
        ("boundary-m65-k3840-n15360", 65, 3840, 15360, False),
        # Production K/N and non-64-aligned M/N representatives.  These are
        # sampled at tile boundaries to keep this host oracle reviewable.
        ("production-m17-k5120-n17408", 17, 5120, 17408, False),
        ("production-m17-k17408-n5120", 17, 17408, 5120, False),
        ("nonaligned-m33-k80-n65", 33, 80, 65, False),
    ]
    report = {
        "schema_version": "phase82-id62-independent-oracle-v1",
        "state": "HOST_ORACLE_COMPLETE",
        "gpu_executed": False,
        "runtime_or_build_executed": False,
        "reference": {
            "kind": "exact_dyadic_nvfp4_block16_dot",
            "activation_scale_code": "deterministic row/block E4M3 pattern 0x28..0x68",
            "weight_scale_codes": ["0x30", "0x38", "0x40", "0x48"],
            "tensor_scale": "3/4",
            "source_contracts": [
                "native/hip/src/matmul_kernel.hip.cpp:2506-2600 (ID59)",
                "native/hip/src/matmul_kernel.hip.cpp:2774-2935 (ID62)",
            ],
            "comparison": "float32 structural candidates against exact real result; bit identity is not required",
        },
        "policy": {
            "classification": "N2_HOLD_against_corrected_ID59",
            "reason": "ID62 retains the equation but uses sequential block accumulation with a larger long-K error bound than corrected ID59; host evidence does not prove dispatch, compiler rounding, or model quality.",
            "adoption": "human_decision_required",
        },
        "shapes": [compare_shape(*shape) for shape in shapes],
    }
    source_paths = [
        Path("scripts/dev/phase82_prefill_id62_oracle.py"),
        Path(".local-artifacts/phase82/id62-oracle/runner.cpp"),
        Path(".local-artifacts/phase82/id62-oracle/compile.command.sh"),
        Path(".local-artifacts/phase79-oracles/prefill-reduction-fix/runner.cpp"),
        Path("native/hip/src/matmul_kernel.hip.cpp"),
    ]
    report["source_sha256"] = {
        str(path): sha256(path) for path in source_paths if path.exists()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "shapes": len(report["shapes"]),
        "all_finite": all(
            result[key]["finite"]
            for result in report["shapes"]
            for key in ("baseline_id59_vs_exact", "candidate_id62_vs_exact",
                        "id62_minus_id59")
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
