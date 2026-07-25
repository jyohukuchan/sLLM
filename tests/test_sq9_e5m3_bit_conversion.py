"""CPU proof for the SQ9_0 E5M3-to-binary16 bit conversion contract.

SQ9_0 stores the IEEE-style fields as ``s:e[4:0]:m[2:0]`` in bits 8..0.
E5M3 and binary16 have the same five-bit exponent and bias (15), so a valid
SQ9_0 code occupies binary16 bits 15..7 without any rebiasing or denormal
normalization.  The implementation under test therefore intentionally has a
single conversion operation: ``code << 7``.
"""

from __future__ import annotations

from fractions import Fraction
import struct
import unittest


SQ9_0_CODE_MASK = 0x1FF
SQ9_0_EXPONENT_MASK = 0x1F
SQ9_0_MANTISSA_MASK = 0x07


def sq9_0_e5m3_to_fp16_bits(code: int) -> int:
    """Return the binary16 bit pattern for one valid SQ9_0 E5M3 code.

    This deliberately rejects out-of-range caller input instead of silently
    truncating it.  Once the nine-bit precondition is met, the conversion is
    exactly one left shift; it has no LUT, rebias, or denormal branch.
    """

    if code < 0 or code > SQ9_0_CODE_MASK:
        raise ValueError(f"SQ9_0 code must fit in nine bits, got {code}")
    return code << 7


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent, 1)
    return Fraction(1, 1 << -exponent)


def _finite_e5m3_value(code: int) -> Fraction:
    """Independent mathematical E5M3 decoder for finite patterns only."""

    sign = -1 if (code >> 8) & 1 else 1
    exponent = (code >> 3) & SQ9_0_EXPONENT_MASK
    mantissa = code & SQ9_0_MANTISSA_MASK
    if exponent == 0:
        # E5M3 subnormals have exponent 1 - bias and a three-bit fraction.
        return sign * Fraction(mantissa, 8) * _power_of_two(-14)
    if exponent == SQ9_0_EXPONENT_MASK:
        raise ValueError("E5M3 exp=31 is not finite")
    return sign * Fraction(8 + mantissa, 8) * _power_of_two(exponent - 15)


def _reference_fp16_bits_from_e5m3_semantics(code: int) -> int:
    """Produce the binary16 result without using the SQ9_0 shift equation."""

    sign_bit = (code >> 8) & 1
    exponent = (code >> 3) & SQ9_0_EXPONENT_MASK
    mantissa = code & SQ9_0_MANTISSA_MASK

    if exponent == SQ9_0_EXPONENT_MASK:
        # SQ9_0 preserves IEEE special encodings.  Keeping the three payload
        # bits in binary16 fraction bits 9..7 also preserves the exact NaN
        # payload contract; do not route NaNs through a host float conversion.
        return (sign_bit << 15) | (SQ9_0_EXPONENT_MASK << 10) | (mantissa << 7)
    if exponent == 0 and mantissa == 0:
        return sign_bit << 15

    finite_value = float(_finite_e5m3_value(code))
    return int.from_bytes(struct.pack("<e", finite_value), "little")


class Sq9E5m3BitConversionTests(unittest.TestCase):
    def test_all_512_signed_e5m3_patterns_match_independent_fp16_semantics(self) -> None:
        for code in range(SQ9_0_CODE_MASK + 1):
            with self.subTest(code=f"0x{code:03x}"):
                actual = sq9_0_e5m3_to_fp16_bits(code)
                expected = _reference_fp16_bits_from_e5m3_semantics(code)
                self.assertEqual(actual, expected)
                # E5M3 has only three fraction bits, so binary16's low seven
                # fraction bits must remain zero for every mapped pattern.
                self.assertEqual(actual & 0x007F, 0)

    def test_special_values_and_denormals_keep_the_specified_bit_behavior(self) -> None:
        for sign in (0, 1):
            with self.subTest(sign=sign, kind="zero"):
                self.assertEqual(sq9_0_e5m3_to_fp16_bits(sign << 8), sign << 15)

            with self.subTest(sign=sign, kind="minimum_subnormal"):
                # E5M3's minimum subnormal is 2**-17, represented by binary16
                # fraction bit 7.  No denormal normalization is permitted.
                self.assertEqual(
                    sq9_0_e5m3_to_fp16_bits((sign << 8) | 0x001),
                    (sign << 15) | 0x0080,
                )

            with self.subTest(sign=sign, kind="minimum_normal"):
                self.assertEqual(
                    sq9_0_e5m3_to_fp16_bits((sign << 8) | (1 << 3)),
                    (sign << 15) | 0x0400,
                )

            with self.subTest(sign=sign, kind="infinity"):
                self.assertEqual(
                    sq9_0_e5m3_to_fp16_bits((sign << 8) | (31 << 3)),
                    (sign << 15) | 0x7C00,
                )

            for mantissa in range(1, 8):
                with self.subTest(sign=sign, kind="nan", mantissa=mantissa):
                    self.assertEqual(
                        sq9_0_e5m3_to_fp16_bits(
                            (sign << 8) | (31 << 3) | mantissa
                        ),
                        (sign << 15) | 0x7C00 | (mantissa << 7),
                    )

        # exp=30,m=7 is the finite maximum.  exp=31 remains reserved and is
        # never used as a finite extension of the range.
        self.assertEqual(sq9_0_e5m3_to_fp16_bits((30 << 3) | 7), 0x7B80)

    def test_out_of_range_input_is_rejected_before_the_shift(self) -> None:
        for code in (-1, 512, 1024):
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    sq9_0_e5m3_to_fp16_bits(code)


if __name__ == "__main__":
    unittest.main()
