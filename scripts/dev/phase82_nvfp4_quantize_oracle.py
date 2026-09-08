#!/usr/bin/env python3
"""Independent NumPy oracle and driver for Phase82 NVFP4 quantization.

The production CUDA/HIP codec is intentionally not imported here.  This
script enumerates BF16 -> float32 -> E4M3FN scale -> E2M1 nibble conversion
with NumPy/Python arithmetic, writes a compact fixture, runs the separately
linked HIP binary for each selector mode, and compares packed/scales bytes.

The runner must be linked against the real native archive and accepts the
exact target as argv[1].  No CPU fallback is accepted by this driver.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INPUT_MAGIC = b"P82NQZ1\0"
OUTPUT_MAGIC = b"P82NQR1\0"
MODES = ("flag0", "wave8", "default", "invalid", "forcebaseline")
TARGETS = ("gfx1030", "gfx1201")
K_CASES = (1, 15, 16, 17, 127, 128, 129, 5120)
M_CASES = (1, 3, 17)
GLOBAL_SCALES = (1.0, 0.5, 2.0, 0.0, -1.0)
SPECIAL_BF16 = np.array(
    [
        0x0000,  # +0
        0x8000,  # -0
        0x3F80,  # +1
        0xBF80,  # -1
        0x4000,  # +2
        0xC000,  # -2
        0x4700,  # +32768, saturation-oriented
        0xC700,  # -32768
        0x7F7F,  # largest finite BF16
        0xFF7F,  # most negative finite BF16
        0x0001,  # smallest positive BF16 subnormal
        0x8001,  # smallest negative BF16 subnormal
        0x3FC0,  # +1.5
        0xC040,  # -3
        0x4120,  # +10
        0xC120,  # -10
    ],
    dtype=np.uint16,
)


@dataclass(frozen=True)
class Case:
    m: int
    k: int
    input_tensor_scale: float
    activation_bits: np.ndarray


def _xorshift32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def make_activation(m: int, k: int, case_index: int) -> np.ndarray:
    count = m * k
    bits = np.empty(count, dtype=np.uint16)
    state = (0x243F6A88 ^ (case_index * 0x9E3779B9)) & 0xFFFFFFFF
    for index in range(count):
        if index < SPECIAL_BF16.size:
            bits[index] = SPECIAL_BF16[(index + case_index) % SPECIAL_BF16.size]
            continue
        state = _xorshift32(state + index + 1)
        sign = (state >> 31) << 15
        exponent = 0x60 + ((state >> 23) % 0x1F)  # finite, varied magnitude
        mantissa = (state >> 8) & 0x7F
        bits[index] = np.uint16(sign | (exponent << 7) | mantissa)
    return bits


def make_cases() -> list[Case]:
    cases: list[Case] = []
    case_index = 0
    for m_index, m in enumerate(M_CASES):
        for k_index, k in enumerate(K_CASES):
            global_scale = GLOBAL_SCALES[(m_index * len(K_CASES) + k_index) % len(GLOBAL_SCALES)]
            cases.append(Case(m, k, global_scale, make_activation(m, k, case_index)))
            case_index += 1
    # Every listed positive magnitude is an E2M1 midpoint. Keep +6 in the
    # same block so its scale is fixed by the requested global scale of one.
    cases.append(
        Case(
            1,
            16,
            1.0,
            np.array(
                [
                    0x0000,
                    0x8000,
                    0x3E80,
                    0xBE80,
                    0x3F40,
                    0xBF40,
                    0x3FA0,
                    0xBFA0,
                    0x3FE0,
                    0xBFE0,
                    0x4020,
                    0xC020,
                    0x4060,
                    0xC060,
                    0x40A0,
                    0x40C0,
                ],
                dtype=np.uint16,
            ),
        )
    )
    cases.append(Case(1, 16, 1.0, np.zeros(16, dtype=np.uint16)))
    return cases


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    widened = bits.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


def e4m3fn_decode(code: int) -> np.float32:
    sign = (code & 0x80) << 24
    magnitude = code & 0x7F
    exponent = magnitude >> 3
    mantissa = magnitude & 0x07
    if exponent == 0:
        value = np.float32(mantissa) * np.float32(2.0**-9)
        if code & 0x80:
            value = np.float32(-value)
        return value
    if magnitude == 0x7F:
        return np.float32(np.nan)
    raw = np.uint32(sign | ((exponent + 120) << 23) | (mantissa << 20))
    return raw.view(np.float32).item()


# Keep the scale encoder independent from the production bit-rounding path.
# The finite positive codebook is small enough to enumerate directly. This
# also makes the tie-even rule explicit at every normal/subnormal boundary.
_E4M3FN_POSITIVE_VALUES = tuple(
    e4m3fn_decode(code) for code in range(0x7F)
)


def e4m3fn_encode(value: np.float32) -> int:
    value = np.float32(value)
    sign = 0x80 if np.signbit(value) else 0
    magnitude = np.float32(abs(value))
    if np.isnan(magnitude):
        return 0x7F
    if magnitude == np.float32(0.0):
        return sign
    if not np.isfinite(magnitude):
        return sign | 0x7E
    # OCP E4M3FN has a finite positive endpoint at code 0x7e (448). Clamp
    # before distance arithmetic: for a large finite BF16 input, even a
    # float64 subtraction from 448 cannot distinguish the input from the
    # endpoint once the input is around 1e38. This is finite nearest-value
    # quantization with saturation, rather than an accidental tie with code 0.
    maximum = _E4M3FN_POSITIVE_VALUES[0x7E]
    if magnitude >= maximum:
        return sign | 0x7E
    selected = 0
    # Keep the distance comparison in Python float64 for the finite range
    # below the endpoint; this preserves nearest/tie-even behavior.
    selected_error = float("inf")
    for code, representable in enumerate(_E4M3FN_POSITIVE_VALUES):
        error = abs(float(magnitude) - float(representable))
        if error < selected_error or (
            error == selected_error
            and (code & 1) == 0
            and (selected & 1) != 0
        ):
            selected = code
            selected_error = error
    return sign | selected


_E2M1_POSITIVE_VALUES = np.array(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=np.float32
)


def e2m1_decode(code: int) -> np.float32:
    value = _E2M1_POSITIVE_VALUES[code & 0x07]
    return np.float32(-value if code & 0x08 else value)


def e2m1_encode(value: np.float32) -> int:
    value = np.float32(value)
    sign = 0x08 if np.signbit(value) else 0
    if np.isnan(value):
        return sign
    magnitude = np.float32(min(abs(value), np.float32(6.0)))
    selected = 0
    selected_error = magnitude
    for code in range(1, 8):
        error = np.float32(abs(np.float32(magnitude - e2m1_decode(code))))
        if error < selected_error or (
            error == selected_error and (code & 1) == 0 and (selected & 1) != 0
        ):
            selected = code
            selected_error = error
    return sign | selected


def oracle_case(case: Case) -> tuple[bytes, bytes]:
    values = bf16_to_float32(case.activation_bits).reshape(case.m, case.k)
    packed_row_bytes = (case.k + 1) // 2
    blocks_per_row = (case.k + 15) // 16
    packed = np.zeros((case.m, packed_row_bytes), dtype=np.uint8)
    scales = np.zeros((case.m, blocks_per_row), dtype=np.uint8)
    global_scale = np.float32(case.input_tensor_scale)
    for row in range(case.m):
        for block in range(blocks_per_row):
            begin = block * 16
            block_values = values[row, begin : min(begin + 16, case.k)]
            maximum = np.max(np.abs(block_values), initial=np.float32(0.0))
            raw_scale = (
                np.float32(0.0)
                if maximum == np.float32(0.0) or not (global_scale > np.float32(0.0))
                else np.float32(maximum / np.float32(np.float32(6.0) * global_scale))
            )
            scale_code = e4m3fn_encode(raw_scale)
            scales[row, block] = np.uint8(scale_code)
            decoded_scale = np.float32(e4m3fn_decode(scale_code) * global_scale)
            for lane in range(16):
                column = begin + lane
                if column >= case.k:
                    break
                code = (
                    e2m1_encode(np.float32(values[row, column] / decoded_scale))
                    if decoded_scale > np.float32(0.0)
                    else 0
                )
                byte_index = column // 2
                if column & 1:
                    packed[row, byte_index] |= np.uint8(code << 4)
                else:
                    packed[row, byte_index] = np.uint8(code)
    return packed.tobytes(), scales.tobytes()


def write_fixture(path: Path, cases: list[Case]) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", INPUT_MAGIC, 1, len(cases)))
        for case in cases:
            stream.write(struct.pack("<QQfQ", case.m, case.k, case.input_tensor_scale, case.m * case.k))
            stream.write(case.activation_bits.astype("<u2", copy=False).tobytes())


def read_result(path: Path, target: str, mode: str, cases: list[Case]) -> list[tuple[bytes, bytes]]:
    data = path.read_bytes()
    offset = 0

    def take(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise AssertionError("truncated runner result")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    magic, version, count = take("<8sII")
    if magic != OUTPUT_MAGIC or version != 1 or count != len(cases):
        raise AssertionError("runner result identity/count mismatch")
    target_field = take("<8s")[0].split(b"\0", 1)[0].decode()
    mode_field = take("<16s")[0].split(b"\0", 1)[0].decode()
    if target_field != target or mode_field != mode:
        raise AssertionError(f"runner identity mismatch: {target_field=} {mode_field=}")
    result = []
    for case in cases:
        m, k, packed_bytes, scale_bytes = take("<QQQQ")
        if (m, k) != (case.m, case.k):
            raise AssertionError("runner case shape mismatch")
        packed = data[offset : offset + packed_bytes]
        offset += packed_bytes
        scales = data[offset : offset + scale_bytes]
        offset += scale_bytes
        if len(packed) != packed_bytes or len(scales) != scale_bytes:
            raise AssertionError("runner result payload truncated")
        result.append((packed, scales))
    if offset != len(data):
        raise AssertionError("runner result has trailing bytes")
    return result


def run_runner(runner: Path, target: str, mode: str, fixture: Path, output: Path) -> None:
    command = [str(runner.expanduser().resolve()), target, mode, str(fixture), str(output)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"runner failed ({target}/{mode}) rc={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if "status=ok" not in completed.stdout:
        raise RuntimeError(f"runner did not report status=ok: {completed.stdout}")


def self_test() -> None:
    cases = make_cases()
    expected = [oracle_case(case) for case in cases]
    if len(cases) != len(M_CASES) * len(K_CASES) + 2:
        raise AssertionError("case matrix incomplete")
    if not any(case.input_tensor_scale <= 0.0 for case in cases):
        raise AssertionError("global-scale edge cases missing")
    if not any(any(bits in SPECIAL_BF16 for bits in case.activation_bits) for case in cases):
        raise AssertionError("special BF16 values missing")
    midpoint = cases[-2]
    if midpoint.input_tensor_scale != 1.0 or np.max(np.abs(bf16_to_float32(midpoint.activation_bits))) != 6.0:
        raise AssertionError("E2M1 midpoint case must use scale=1 and maximum=6")
    if not np.array_equal(cases[-1].activation_bits, np.zeros(16, dtype=np.uint16)):
        raise AssertionError("all-zero case missing")
    # Small and endpoint checks pin the oracle's mathematical contract and
    # catch accidental overflow/tie behavior independently of the GPU runner.
    if e4m3fn_encode(np.float32(0.0)) != 0x00:
        raise AssertionError("E4M3FN zero encoding changed")
    if e4m3fn_encode(np.float32(448.0)) != 0x7E:
        raise AssertionError("E4M3FN finite endpoint encoding changed")
    if e4m3fn_encode(np.float32(449.0)) != 0x7E:
        raise AssertionError("E4M3FN finite endpoint saturation missing")
    e4_tie = np.float32(
        (float(e4m3fn_decode(0x01)) + float(e4m3fn_decode(0x02))) / 2.0
    )
    if e4m3fn_encode(e4_tie) != 0x02:
        raise AssertionError("E4M3FN tie-even encoding changed")
    max_bf16 = bf16_to_float32(np.array([0x7F7F], dtype=np.uint16))[0]
    if e4m3fn_encode(max_bf16) != 0x7E:
        raise AssertionError("maximum finite BF16 scale must saturate E4M3FN")
    if e2m1_encode(np.float32(0.0)) != 0x00:
        raise AssertionError("E2M1 zero encoding changed")
    if e2m1_encode(np.float32(6.0)) != 0x07:
        raise AssertionError("E2M1 finite endpoint encoding changed")
    if e2m1_encode(np.float32(7.0)) != 0x07:
        raise AssertionError("E2M1 finite endpoint saturation missing")
    if e2m1_encode(np.float32(0.75)) != 0x02:
        raise AssertionError("E2M1 tie-even encoding changed")
    for case, (packed, scales) in zip(cases, expected):
        if len(packed) != case.m * ((case.k + 1) // 2):
            raise AssertionError("packed size mismatch")
        if len(scales) != case.m * ((case.k + 15) // 16):
            raise AssertionError("scale size mismatch")
    print(f"oracle self-test: cases={len(cases)} status=ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.runner is None:
        return 0 if args.self_test else parser.error("--runner is required")
    if args.target is None:
        return parser.error("--target is required with --runner")
    runner = args.runner.expanduser().resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        return parser.error(f"runner is not executable: {runner}")

    cases = make_cases()
    expected = [oracle_case(case) for case in cases]
    with tempfile.TemporaryDirectory(prefix="phase82-nvfp4-") as directory:
        root = Path(directory)
        fixture = root / "fixture.bin"
        write_fixture(fixture, cases)
        baseline: list[tuple[bytes, bytes]] | None = None
        for mode in MODES:
            output = root / f"{args.target}-{mode}.bin"
            run_runner(runner, args.target, mode, fixture, output)
            actual = read_result(output, args.target, mode, cases)
            for case, (wanted_packed, wanted_scales), (got_packed, got_scales) in zip(
                cases, expected, actual
            ):
                if got_packed != wanted_packed or got_scales != wanted_scales:
                    raise AssertionError(
                        f"oracle mismatch target={args.target} mode={mode} "
                        f"shape=({case.m},{case.k})"
                    )
            if baseline is None:
                baseline = actual
            elif actual != baseline:
                raise AssertionError(f"selector mode changed bytes: {args.target}/{mode}")
            print(f"oracle compare: target={args.target} mode={mode} cases={len(cases)} status=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError) as error:
        print(f"phase82 nvfp4 oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
