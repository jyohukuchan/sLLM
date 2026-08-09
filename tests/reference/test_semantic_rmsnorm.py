from __future__ import annotations

import numpy as np
import pytest

from tests.reference.semantic_rmsnorm import (
    MAX_SEMANTIC_ELEMENTS,
    MAX_SEMANTIC_N,
    SEMANTIC_RMSNORM_OFFSET_ONE,
    SemanticRmsNormShape,
    assert_semantic_outputs_close,
    classify_semantic_output,
    compare_semantic_outputs,
    semantic_rmsnorm,
    validate_semantic_rmsnorm_shapes,
)


pytestmark = pytest.mark.tier_h2


ACCEPTED_N = (
    1,
    3,
    17,
    37,
    73,
    255,
    256,
    257,
    1023,
    1024,
    1025,
    2559,
    2560,
    2561,
    4095,
    4096,
)


def _zero_inputs(shape: tuple[int, ...], features: int) -> tuple[np.ndarray, np.ndarray]:
    activation = np.zeros(shape, dtype=np.uint16)
    raw_scale = np.zeros((features,), dtype=np.uint16)
    return activation, raw_scale


def test_semantic_acceptance_n_values_and_offset_one_output_storage() -> None:
    for features in ACCEPTED_N:
        activation, raw_scale = _zero_inputs((1, features), features)
        activation.reshape(-1)[0] = 0x3F80  # BF16 1.0
        raw_scale[0] = 0x3F80  # effective scale is 2.0

        result = semantic_rmsnorm(
            activation,
            raw_scale,
            1.0e-6,
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )

        assert result.dtype == np.uint16
        assert result.shape == (1, features)
        assert result.size <= MAX_SEMANTIC_ELEMENTS
        assert classify_semantic_output(result)[0, 0] == "finite"

    activation, raw_scale = _zero_inputs((2, 2560), 2560)
    result = semantic_rmsnorm(activation, raw_scale, 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)
    assert result.shape == (2, 2560)


def test_semantic_flattens_rank_one_two_three_leading_dimensions() -> None:
    for shape, features, rows in (
        ((3,), 3, 1),
        ((2, 3), 3, 2),
        ((1, 2, 3), 3, 2),
    ):
        contract = validate_semantic_rmsnorm_shapes(shape, (features,))
        assert isinstance(contract, SemanticRmsNormShape)
        assert contract.rows == rows
        assert contract.elements == rows * features

        activation, raw_scale = _zero_inputs(shape, features)
        result = semantic_rmsnorm(activation, raw_scale, 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)
        assert result.shape == shape


def test_semantic_accepts_both_exact_maximum_row_feature_products() -> None:
    for shape in ((1024, 256), (64, 4096)):
        contract = validate_semantic_rmsnorm_shapes(shape, (shape[-1],))
        assert contract.rows * contract.features == MAX_SEMANTIC_ELEMENTS
        activation, raw_scale = _zero_inputs(shape, shape[-1])
        result = semantic_rmsnorm(activation, raw_scale, 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)
        assert result.shape == shape
        assert result.size == MAX_SEMANTIC_ELEMENTS


def test_semantic_rejects_n_and_product_before_materializing_inputs() -> None:
    # Shape validation is allocation-free. These rejected shapes are never
    # passed to np.zeros or to the semantic computation.
    for activation_shape, raw_shape in (
        ((1, 4097), (4097,)),
        ((1025, 256), (256,)),
        ((65, 4096), (4096,)),
        ((2**200, 2**200, 1), (1,)),
    ):
        with pytest.raises(ValueError):
            validate_semantic_rmsnorm_shapes(activation_shape, raw_shape)

    assert MAX_SEMANTIC_N == 4096


def test_semantic_rejects_huge_boolean_and_non_integral_dimensions() -> None:
    with pytest.raises(ValueError, match="RMSNorm R\\*N"):
        validate_semantic_rmsnorm_shapes((2**4096, 1), (1,))

    for activation_shape, raw_scale_shape in (
        ((True, 3), (3,)),
        ((np.bool_(True), 3), (3,)),
        ((1.0, 3), (3,)),
        ((2, 3), (3.0,)),
    ):
        with pytest.raises(ValueError, match="dimensions"):
            validate_semantic_rmsnorm_shapes(activation_shape, raw_scale_shape)  # type: ignore[arg-type]


def test_semantic_rejects_empty_zero_and_wrong_rank_or_length_shapes() -> None:
    invalid_shapes = (
        ((), (1,)),
        ((0,), (1,)),
        ((0, 3), (3,)),
        ((2, 0), (0,)),
        ((2, 3), ()),
        ((2, 3), (1, 3)),
        ((2, 3), (2,)),
        ((2, 3), (4,)),
    )
    for activation_shape, raw_shape in invalid_shapes:
        with pytest.raises(ValueError):
            validate_semantic_rmsnorm_shapes(activation_shape, raw_shape)

    activation, raw_scale = _zero_inputs((2, 3), 3)
    with pytest.raises(ValueError, match="rank one"):
        semantic_rmsnorm(activation, raw_scale.reshape(1, 3), 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)
    with pytest.raises(ValueError, match="length"):
        semantic_rmsnorm(activation, raw_scale[:2], 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)


def test_semantic_accepts_ndarray_and_rectangular_list_tuple_inputs() -> None:
    activation_list = [[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]]
    raw_scale_tuple = (0.0, 0.25, -0.25)
    list_result = semantic_rmsnorm(
        activation_list,
        raw_scale_tuple,
        1.0e-6,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )

    ndarray_result = semantic_rmsnorm(
        np.asarray(activation_list, dtype=np.float32),
        np.asarray(raw_scale_tuple, dtype=np.float32),
        1.0e-6,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )
    np.testing.assert_array_equal(list_result, ndarray_result)


class _ArrayLikeSpy:
    shape = (1, 3)

    def __init__(self) -> None:
        self.array_calls = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        self.array_calls += 1
        raise AssertionError("the semantic oracle must reject this object first")


class _SpoofedShapeNdarray(np.ndarray):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, 3)


class _ArrayInterfaceList(list):
    @property
    def __array_interface__(self) -> object:
        raise AssertionError("the semantic oracle must not inspect list hooks")


def test_semantic_rejects_arbitrary_shape_array_objects_before_conversion() -> None:
    activation_spy = _ArrayLikeSpy()
    raw_scale = np.zeros((3,), dtype=np.uint16)
    with pytest.raises(ValueError, match="ndarray or recursively rectangular"):
        semantic_rmsnorm(
            activation_spy,
            raw_scale,
            1.0e-6,
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )
    assert activation_spy.array_calls == 0


def test_semantic_rejects_ndarray_subclass_before_spoofed_shape_acceptance() -> None:
    activation = np.zeros((4097,), dtype=np.uint16).view(_SpoofedShapeNdarray)
    raw_scale = np.zeros((3,), dtype=np.uint16)

    with pytest.raises(ValueError, match="ndarray or recursively rectangular"):
        semantic_rmsnorm(
            activation,
            raw_scale,
            1.0e-6,
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )


def test_semantic_rejects_list_subclass_without_invoking_array_interface() -> None:
    activation = _ArrayInterfaceList([[0.0, 0.0, 0.0]])
    raw_scale = np.zeros((3,), dtype=np.uint16)

    with pytest.raises(ValueError, match="ndarray or recursively rectangular"):
        semantic_rmsnorm(
            activation,
            raw_scale,
            1.0e-6,
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )


def test_semantic_requires_explicit_epsilon_and_scale_mode() -> None:
    activation, raw_scale = _zero_inputs((1, 3), 3)
    with pytest.raises(TypeError):
        semantic_rmsnorm(activation, raw_scale, 1.0e-6)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        semantic_rmsnorm(activation, raw_scale, scale_mode=SEMANTIC_RMSNORM_OFFSET_ONE)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="offset_one"):
        semantic_rmsnorm(activation, raw_scale, 1.0e-6, "ordinary")


def test_semantic_rejects_nonpositive_or_nonfinite_or_unrepresentable_epsilon() -> None:
    activation, raw_scale = _zero_inputs((1, 3), 3)
    for epsilon in (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        2**4096,
        np.float64(1.0e-50),
    ):
        with pytest.raises(ValueError, match="epsilon"):
            semantic_rmsnorm(activation, raw_scale, epsilon, SEMANTIC_RMSNORM_OFFSET_ONE)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        semantic_rmsnorm(
            activation,
            raw_scale,
            np.asarray([1.0e-6]),
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="BF16"):
        semantic_rmsnorm(
            np.asarray([[1, 2, 3]], dtype=np.int32),
            raw_scale,
            1.0e-6,
            SEMANTIC_RMSNORM_OFFSET_ONE,
        )


def test_semantic_nonfinite_bf16_activation_and_scale_propagate_without_scan() -> None:
    activation = np.asarray([[0x7F80, 0x3F80, 0xFF80, 0x7FC1]], dtype=np.uint16)
    raw_scale = np.zeros((4,), dtype=np.uint16)
    activation_result = semantic_rmsnorm(
        activation,
        raw_scale,
        1.0e-6,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )
    np.testing.assert_array_equal(
        classify_semantic_output(activation_result),
        np.asarray([["NaN", "NaN", "NaN", "NaN"]]),
    )

    finite_activation = np.asarray([[0x3F80, 0x4000, 0x4040]], dtype=np.uint16)
    nonfinite_scale = np.asarray([0x7F80, 0xFF80, 0x7FC1], dtype=np.uint16)
    scale_result = semantic_rmsnorm(
        finite_activation,
        nonfinite_scale,
        1.0e-6,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )
    np.testing.assert_array_equal(
        classify_semantic_output(scale_result),
        np.asarray([["+Inf", "-Inf", "NaN"]]),
    )


def test_semantic_bf16_nan_payload_and_sign_are_preserved_as_nan_class() -> None:
    activation = np.asarray([[0x7FC1, 0xFFC2, 0x7F80]], dtype=np.uint16)
    raw_scale = np.zeros((3,), dtype=np.uint16)
    result = semantic_rmsnorm(activation, raw_scale, 1.0e-6, SEMANTIC_RMSNORM_OFFSET_ONE)
    assert np.all(classify_semantic_output(result) == "NaN")


def test_semantic_comparator_passes_finite_nonfinite_and_ignores_nan_payload() -> None:
    reference = np.asarray([0x7FC1, 0x7F80, 0xFF80, 0x3F80], dtype=np.uint16)
    actual = np.asarray([0xFFC2, 0x7F80, 0xFF80, 0x3F80], dtype=np.uint16)
    comparison = compare_semantic_outputs(actual, reference, atol=0.0, rtol=0.0)
    assert comparison.passed
    assert comparison.class_mismatch_count == 0
    assert comparison.finite_mismatch_count == 0

    assert not compare_semantic_outputs(
        np.asarray([0xFF80], dtype=np.uint16),
        np.asarray([0x7F80], dtype=np.uint16),
        atol=0.0,
        rtol=0.0,
    )
    assert not compare_semantic_outputs(
        np.asarray([0x7FC1], dtype=np.uint16),
        np.asarray([0x3F80], dtype=np.uint16),
        atol=0.0,
        rtol=0.0,
    )


def test_semantic_comparator_uses_supplied_bounded_finite_tolerance() -> None:
    reference = np.asarray([2.0], dtype=np.float32)
    passing = compare_semantic_outputs(
        np.asarray([2.11], dtype=np.float32),
        reference,
        atol=0.01,
        rtol=0.05,
    )
    assert passing
    assert passing.max_abs_error is not None

    failing = compare_semantic_outputs(
        np.asarray([2.12], dtype=np.float32),
        reference,
        atol=0.01,
        rtol=0.05,
    )
    assert not failing
    assert failing.finite_mismatch_count == 1
    with pytest.raises(AssertionError, match="finite tolerance"):
        assert_semantic_outputs_close(
            np.asarray([2.12], dtype=np.float32),
            reference,
            atol=0.01,
            rtol=0.05,
        )


def test_semantic_comparator_preserves_float_dtype_for_finite_math_and_classification() -> None:
    float32_regression = compare_semantic_outputs(
        np.asarray([-np.finfo(np.float32).max], dtype=np.float32),
        np.asarray([1.0e38], dtype=np.float32),
        atol=0.0,
        rtol=3.5,
    )
    assert not float32_regression
    assert float32_regression.finite_mismatch_count == 1

    float64_classification = compare_semantic_outputs(
        np.asarray([1.0e100], dtype=np.float64),
        np.asarray([np.inf], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert not float64_classification
    assert float64_classification.class_mismatch_count == 1
    assert classify_semantic_output(np.asarray([1.0e100], dtype=np.float64))[0] == "finite"


def test_semantic_comparator_rejects_float64_difference_and_bound_overflow() -> None:
    maximum = np.finfo(np.float64).max

    comparison = compare_semantic_outputs(
        np.asarray([maximum], dtype=np.float64),
        np.asarray([-maximum], dtype=np.float64),
        atol=0.0,
        rtol=1.5,
    )

    assert not comparison
    assert comparison.finite_mismatch_count == 1
    assert comparison.max_abs_error is not None
    assert np.isinf(comparison.max_abs_error)


def test_semantic_comparator_rejects_shape_and_tolerance_errors() -> None:
    shape_result = compare_semantic_outputs(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        atol=0.0,
        rtol=0.0,
    )
    assert not shape_result
    assert "shape mismatch" in shape_result.reason
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_semantic_outputs_close(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
            atol=0.0,
            rtol=0.0,
        )

    for bad_tolerance in (-1.0, float("nan"), float("inf"), True, np.asarray([0.0])):
        with pytest.raises(ValueError, match="atol"):
            compare_semantic_outputs(
                np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                atol=bad_tolerance,  # type: ignore[arg-type]
                rtol=0.0,
            )
    for bad_tolerance in (-1.0, float("nan"), float("inf"), False, np.asarray([0.0])):
        with pytest.raises(ValueError, match="rtol"):
            compare_semantic_outputs(
                np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                atol=0.0,
                rtol=bad_tolerance,  # type: ignore[arg-type]
            )


def test_semantic_rmsnorm_exact_bf16_formula_and_rne_output_code_kat() -> None:
    # The inputs and expected output are BF16 storage codes.  For the first
    # row, sum(x*x)=14 and the denominator is 14/3 + 1/2.  The second row has
    # sum(x*x)=21/4 and the same epsilon.  Scale is explicitly 1 + raw_scale.
    activation = np.asarray(
        [
            [[0x3F80, 0xC000, 0x4040], [0x3F00, 0x4000, 0xBF80]],
        ],
        dtype=np.uint16,
    )
    raw_scale = np.asarray([0x3F80, 0x0000, 0xBF80], dtype=np.uint16)
    expected_codes = np.asarray(
        [[[0x3F61, 0xBF61, 0x0000], [0x3F2B, 0x3FAB, 0x8000]]],
        dtype=np.uint16,
    )

    result = semantic_rmsnorm(
        activation,
        raw_scale,
        0.5,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )
    np.testing.assert_array_equal(result, expected_codes)
    assert_semantic_outputs_close(result, expected_codes, atol=0.0, rtol=0.0)


def test_semantic_rmsnorm_discriminating_fp32_accumulation_kat() -> None:
    activation = np.asarray([[0x4380, 0x3F80, 0x0000]], dtype=np.uint16)
    raw_scale = np.zeros((3,), dtype=np.uint16)

    # FP16 accumulation overflows and yields zeros, so this KAT guards explicit FP32 accumulation.
    result = semantic_rmsnorm(
        activation,
        raw_scale,
        1.0e-6,
        SEMANTIC_RMSNORM_OFFSET_ONE,
    )

    np.testing.assert_array_equal(
        result,
        np.asarray([[0x3FDE, 0x3BDE, 0x0000]], dtype=np.uint16),
    )
