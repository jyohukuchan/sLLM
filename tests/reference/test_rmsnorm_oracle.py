from __future__ import annotations

import numpy as np
import pytest

from tests.reference.oracles import (
    MAX_ORACLE_ELEMENTS,
    RMSNORM_OFFSET_ONE,
    bf16_decode,
    bf16_encode_rne,
    rmsnorm_bf16,
)


def _float32_from_bits(bits: list[int]) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _expected_rmsnorm(activation: np.ndarray, raw_scale: np.ndarray, epsilon: float) -> np.ndarray:
    x = bf16_decode(activation).astype(np.float32, copy=False)
    raw = bf16_decode(raw_scale).astype(np.float32, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sum_squares = np.sum(np.multiply(x, x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
        mean_square = np.divide(sum_squares, np.float32(x.shape[-1]), dtype=np.float32)
        inverse_rms = np.reciprocal(
            np.sqrt(np.add(mean_square, np.float32(epsilon), dtype=np.float32), dtype=np.float32),
            dtype=np.float32,
        )
        expected_values = np.multiply(
            np.multiply(x, inverse_rms, dtype=np.float32),
            np.add(np.float32(1.0), raw, dtype=np.float32),
            dtype=np.float32,
        )
    return bf16_encode_rne(expected_values)


@pytest.mark.tier_h2
def test_bf16_rne_kats_cover_ties_signed_zero_finite_subnormal_nan_and_inf() -> None:
    values = _float32_from_bits(
        [
            0x3F808000,  # halfway, even upper BF16 code: round down
            0x3F818000,  # halfway, odd upper BF16 code: round up
            0x00008000,  # smallest-subnormal halfway, even zero: round down
            0x00018000,  # smallest-subnormal halfway, odd one: round up
            0x00010000,  # finite BF16 subnormal
            0x00000000,  # +0
            0x80000000,  # -0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC10000,  # nonzero NaN payload
        ]
    )
    encoded = bf16_encode_rne(values)
    np.testing.assert_array_equal(
        encoded,
        np.asarray(
            [0x3F80, 0x3F82, 0x0000, 0x0002, 0x0001, 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1],
            dtype=np.uint16,
        ),
    )

    decoded = bf16_decode(encoded)
    assert decoded.dtype == np.float32
    assert decoded[5] == 0.0 and not np.signbit(decoded[5])
    assert decoded[6] == 0.0 and np.signbit(decoded[6])
    assert decoded[4] > 0.0 and decoded[4] < np.finfo(np.float32).tiny
    assert np.isposinf(decoded[7])
    assert np.isneginf(decoded[8])
    assert np.isnan(decoded[9])


@pytest.mark.tier_h2
def test_rmsnorm_bf16_covers_requested_last_dimensions_one_and_multiple_rows() -> None:
    epsilon = 1.0e-6
    for last_dimension in (1, 3, 17, 255, 256, 257, 2560):
        rows = 1 if last_dimension in (1, 2560) else 2
        shape = (last_dimension,) if rows == 1 else (rows, last_dimension)
        values = np.zeros(shape, dtype=np.float32)
        values.flat[0] = 1.0e-3
        values.flat[-1] = 3.0e4
        if values.size > 2:
            values.flat[values.size // 2] = -2.0
        raw_values = np.linspace(-0.5, 0.5, last_dimension, dtype=np.float32)
        activation = bf16_encode_rne(values)
        raw_scale = bf16_encode_rne(raw_values)

        result = rmsnorm_bf16(activation, raw_scale, epsilon, RMSNORM_OFFSET_ONE)
        np.testing.assert_array_equal(result, _expected_rmsnorm(activation, raw_scale, epsilon))
        assert result.dtype == np.uint16
        assert result.shape == shape
        assert np.isfinite(bf16_decode(result)).all()


@pytest.mark.tier_h2
def test_rmsnorm_bf16_is_fp32_accumulating_offset_one_and_preserves_zero() -> None:
    activation = bf16_encode_rne(np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    raw_scale = bf16_encode_rne(np.asarray([0.0, 1.0, -0.5], dtype=np.float32))
    result = rmsnorm_bf16(activation, raw_scale, 1.0e-6, RMSNORM_OFFSET_ONE)

    np.testing.assert_array_equal(result, _expected_rmsnorm(activation, raw_scale, 1.0e-6))
    # Compact exact-code KAT: this independently fixes the FP32 formula and
    # BF16 RNE result for the nonzero row, while the second row proves zero
    # stays signed-positive under the same offset-one path.
    np.testing.assert_array_equal(
        result,
        np.asarray(
            [[0x3EED, 0x3FED, 0x3F32], [0x0000, 0x0000, 0x0000]],
            dtype=np.uint16,
        ),
    )
    np.testing.assert_array_equal(result[1], np.asarray([0, 0, 0], dtype=np.uint16))

    conventional_scale = bf16_encode_rne(np.zeros(3, dtype=np.float32))
    conventional = rmsnorm_bf16(activation, conventional_scale, 1.0e-6, RMSNORM_OFFSET_ONE)
    assert not np.array_equal(result[0], conventional[0])


@pytest.mark.tier_h2
def test_rmsnorm_bf16_discriminating_fp32_accumulation_kat() -> None:
    activation = np.asarray([[0x4380, 0x3F80, 0x0000]], dtype=np.uint16)
    raw_scale = np.zeros((3,), dtype=np.uint16)

    result = rmsnorm_bf16(activation, raw_scale, 1.0e-6, RMSNORM_OFFSET_ONE)

    np.testing.assert_array_equal(
        result,
        np.asarray([[0x3FDE, 0x3BDE, 0x0000]], dtype=np.uint16),
    )


@pytest.mark.tier_h2
def test_rmsnorm_bf16_rejects_invalid_epsilon_shapes_and_keeps_h2_bound() -> None:
    activation = bf16_encode_rne(np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32))
    raw_scale = bf16_encode_rne(np.asarray([0.0, 0.25, -0.25], dtype=np.float32))

    for epsilon in (0.0, -1.0, float("nan"), float("inf"), float("-inf"), 2**4096):
        with pytest.raises(ValueError, match="epsilon"):
            rmsnorm_bf16(activation, raw_scale, epsilon, RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="offset_one"):
        rmsnorm_bf16(activation, raw_scale, 1.0e-6, "ordinary")

    with pytest.raises(ValueError, match="rank"):
        rmsnorm_bf16(activation[0, 0], raw_scale, 1.0e-6, RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="non-empty"):
        rmsnorm_bf16(np.empty((0, 3), dtype=np.uint16), raw_scale, 1.0e-6, RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="rank one"):
        rmsnorm_bf16(activation, raw_scale.reshape(1, 3), 1.0e-6, RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="last dimension"):
        rmsnorm_bf16(activation, raw_scale[:2], 1.0e-6, RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="tiny oracle"):
        rmsnorm_bf16(
            np.zeros(MAX_ORACLE_ELEMENTS + 1, dtype=np.uint16),
            np.zeros(1, dtype=np.uint16),
            1.0e-6,
            RMSNORM_OFFSET_ONE,
        )
    with pytest.raises(ValueError, match="tiny oracle"):
        rmsnorm_bf16(
            np.zeros((2, 2560), dtype=np.uint16),
            np.zeros(2560, dtype=np.uint16),
            1.0e-6,
            RMSNORM_OFFSET_ONE,
        )

    for nonfinite_activation in (
        np.asarray([[np.nan, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[np.inf, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[-np.inf, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[0x7FC1, 0x0000, 0x3F80]], dtype=np.uint16),
        np.asarray([[0x7F80, 0x0000, 0x3F80]], dtype=np.uint16),
    ):
        result = rmsnorm_bf16(nonfinite_activation, raw_scale, 1.0e-6, RMSNORM_OFFSET_ONE)
        assert np.isnan(bf16_decode(result)[0, 0])

    for nonfinite_scale, expected in (
        (np.asarray([0x3F80, 0x7FC1, 0x3F80], dtype=np.uint16), "nan"),
        (np.asarray([0x3F80, 0x7F80, 0x3F80], dtype=np.uint16), "neginf"),
        (np.asarray([0x3F80, 0xFF80, 0x3F80], dtype=np.uint16), "posinf"),
    ):
        result = bf16_decode(rmsnorm_bf16(activation, nonfinite_scale, 1.0e-6, RMSNORM_OFFSET_ONE))
        if expected == "nan":
            assert np.isnan(result[0, 1])
        elif expected == "posinf":
            assert np.isposinf(result[0, 1])
        else:
            assert np.isneginf(result[0, 1])
