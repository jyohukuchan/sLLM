"""Bounded semantic G1 RMSNorm oracle and output comparator.

This module is deliberately separate from the H2 oracle in ``oracles.py``.
The semantic slice permits the declared G1 shape bound, propagates nonfinite
BF16 values through float32 IEEE arithmetic, and stores results as BF16 RNE
bits.  Shape validation accepts shapes before any output-sized allocation is
made; callers generating inputs should invoke it before materializing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from tests.reference.oracles import bf16_decode, bf16_encode_rne


SEMANTIC_RMSNORM_MAX_N = 4096
SEMANTIC_RMSNORM_MAX_ELEMENTS = 262_144
SEMANTIC_RMSNORM_OFFSET_ONE = "offset_one"

# Short names make the allocation contract convenient to use in case tests.
MAX_SEMANTIC_N = SEMANTIC_RMSNORM_MAX_N
MAX_SEMANTIC_ELEMENTS = SEMANTIC_RMSNORM_MAX_ELEMENTS
RMSNORM_OFFSET_ONE = SEMANTIC_RMSNORM_OFFSET_ONE


def _shape_tuple(value: Sequence[Integral], name: str) -> tuple[int, ...]:
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a shape sequence") from exc
    if not dimensions:
        raise ValueError(f"{name} must have rank at least one")

    normalized: list[int] = []
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise ValueError(f"{name} dimensions must be Python integers")
        dimension_value = int(dimension)
        if dimension_value < 0:
            raise ValueError(f"{name} dimensions must be non-negative")
        normalized.append(dimension_value)
    return tuple(normalized)


_PYTHON_SCALAR_TYPES = (bool, int, float, complex, str, bytes)


def _shape_without_materializing(value: Any, name: str) -> tuple[Any, ...]:
    """Infer a recursively rectangular Python list/tuple shape."""

    if type(value) not in (list, tuple):
        if isinstance(value, np.ndarray):
            raise ValueError(f"{name} list/tuple must contain scalar values")
        # Only scalar values known to be safe for np.asarray are accepted as
        # leaves. This rejects arbitrary array/shape protocol objects without
        # inspecting their attributes, so no user-defined hook can run.
        if isinstance(value, np.generic) or type(value) in _PYTHON_SCALAR_TYPES:
            return ()
        raise ValueError(
            f"{name} must be a NumPy ndarray or recursively rectangular list/tuple"
        )
    if not value:
        return (0,)
    child_shape = _shape_without_materializing(value[0], name)
    for child in value[1:]:
        if _shape_without_materializing(child, name) != child_shape:
            raise ValueError(f"{name} must be rectangular")
    return (len(value),) + child_shape


def _semantic_input_shape(value: Any, name: str) -> tuple[Any, ...]:
    """Inspect only supported containers, without invoking array conversion."""

    if type(value) is np.ndarray:
        return tuple(value.shape)
    if type(value) in (list, tuple):
        return _shape_without_materializing(value, name)
    raise ValueError(
        f"{name} must be a NumPy ndarray or recursively rectangular list/tuple"
    )


def _array_after_shape_check(value: Any, name: str) -> np.ndarray:
    """Convert an already shape-checked supported input to an ndarray."""

    if type(value) is np.ndarray:
        return value
    if type(value) in (list, tuple):
        return np.asarray(value)
    raise ValueError(
        f"{name} must be a NumPy ndarray or recursively rectangular list/tuple"
    )


@dataclass(frozen=True)
class SemanticRmsNormShape:
    activation_shape: tuple[int, ...]
    raw_scale_shape: tuple[int, ...]
    rows: int
    features: int
    elements: int


def validate_semantic_rmsnorm_shapes(
    activation_shape: Sequence[Integral],
    raw_scale_shape: Sequence[Integral],
) -> SemanticRmsNormShape:
    """Validate shapes using exact Python-integer arithmetic.

    This function performs no array conversion and no allocation.  In
    particular, a caller can reject ``N=4097`` or an over-bound product before
    constructing activation, output, or model-sized storage.
    """

    activation = _shape_tuple(activation_shape, "activation shape")
    raw_scale = _shape_tuple(raw_scale_shape, "raw-scale shape")
    features = activation[-1]
    if features < 1 or features > SEMANTIC_RMSNORM_MAX_N:
        raise ValueError(
            f"RMSNorm N must be in 1..{SEMANTIC_RMSNORM_MAX_N}"
        )

    # math.prod is not used on NumPy scalars: all dimensions above are first
    # converted to ordinary Python ints, so this product cannot wrap.
    rows = 1
    for dimension in activation[:-1]:
        rows *= dimension
    if rows < 1:
        raise ValueError("RMSNorm leading dimensions must flatten to R >= 1")
    elements = rows * features
    if elements > SEMANTIC_RMSNORM_MAX_ELEMENTS:
        raise ValueError(
            "RMSNorm R*N exceeds the semantic G1 element bound "
            f"{SEMANTIC_RMSNORM_MAX_ELEMENTS}"
        )

    if len(raw_scale) != 1:
        raise ValueError("RMSNorm raw scale must have rank one")
    if raw_scale[0] != features:
        raise ValueError("RMSNorm raw scale length must match N")

    return SemanticRmsNormShape(activation, raw_scale, rows, features, elements)


def _validate_epsilon(epsilon: Real) -> np.float32:
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError("RMSNorm epsilon must be a finite positive scalar")
    try:
        scalar = float(epsilon)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("RMSNorm epsilon must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("RMSNorm epsilon must be a finite positive scalar")
    value = np.float32(scalar)
    if not np.isfinite(value) or value <= np.float32(0.0):
        raise ValueError("RMSNorm epsilon must be finite and representable in float32")
    return value


def _bf16_codes(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == np.uint16:
        return raw
    if np.issubdtype(raw.dtype, np.floating):
        # There is intentionally no finite scan here.  NaN and both infinity
        # signs are valid BF16 value evidence for this semantic oracle.
        return bf16_encode_rne(np.asarray(raw, dtype=np.float32))
    raise ValueError(f"{name} must be BF16 uint16 bits or floating-point values")


def semantic_rmsnorm(
    activation: Any,
    raw_scale: Any,
    epsilon: Real,
    scale_mode: str,
) -> np.ndarray:
    """Compute offset-one BF16 RMSNorm under the bounded semantic contract."""

    if scale_mode != SEMANTIC_RMSNORM_OFFSET_ONE:
        raise ValueError("RMSNorm scale_mode must be the explicit offset_one mode")
    epsilon_value = _validate_epsilon(epsilon)

    # Inspect existing ndarray shapes or Python sequence structure before
    # converting values.  Thus an invalid N or product is rejected before this
    # function allocates an activation-sized conversion or output.
    activation_shape = _semantic_input_shape(activation, "activation")
    raw_scale_shape = _semantic_input_shape(raw_scale, "raw scale")
    contract = validate_semantic_rmsnorm_shapes(activation_shape, raw_scale_shape)
    activation_array = _array_after_shape_check(activation, "activation")
    raw_scale_array = _array_after_shape_check(raw_scale, "raw scale")
    activation_codes = _bf16_codes(activation_array, "activation")
    raw_scale_codes = _bf16_codes(raw_scale_array, "raw scale")

    x = bf16_decode(activation_codes).astype(np.float32, copy=False).reshape(
        contract.rows,
        contract.features,
    )
    raw = bf16_decode(raw_scale_codes).astype(np.float32, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        squared = np.multiply(x, x, dtype=np.float32)
        sum_squares = np.sum(squared, axis=1, keepdims=True, dtype=np.float32)
        mean_square = np.divide(
            sum_squares,
            np.float32(contract.features),
            dtype=np.float32,
        )
        denominator = np.add(mean_square, epsilon_value, dtype=np.float32)
        inverse_rms = np.reciprocal(
            np.sqrt(denominator, dtype=np.float32),
            dtype=np.float32,
        )
        effective_scale = np.add(np.float32(1.0), raw, dtype=np.float32)
        normalized = np.multiply(
            np.multiply(x, inverse_rms, dtype=np.float32),
            effective_scale,
            dtype=np.float32,
        )
    return bf16_encode_rne(normalized).reshape(contract.activation_shape)


def semantic_rmsnorm_values(
    activation: Any,
    raw_scale: Any,
    epsilon: Real,
    scale_mode: str,
) -> np.ndarray:
    """Return semantic RMSNorm output decoded to float32 values."""

    return bf16_decode(semantic_rmsnorm(activation, raw_scale, epsilon, scale_mode))


def _output_values(value: Any, name: str) -> np.ndarray:
    _semantic_input_shape(value, name)
    raw = _array_after_shape_check(value, name)
    if raw.dtype == np.uint16:
        return bf16_decode(raw)
    if np.issubdtype(raw.dtype, np.floating):
        # Keep the caller's floating dtype so classification is not changed
        # by a lossy float32 conversion (notably for large float64 values).
        return raw
    raise ValueError(f"{name} must be BF16 uint16 bits or floating-point values")


def classify_semantic_output(value: Any) -> np.ndarray:
    """Classify each output as ``finite``, ``NaN``, ``+Inf``, or ``-Inf``."""

    values = _output_values(value, "output")
    return np.select(
        [np.isnan(values), np.isposinf(values), np.isneginf(values)],
        ["NaN", "+Inf", "-Inf"],
        default="finite",
    )


@dataclass(frozen=True)
class SemanticComparison:
    passed: bool
    reason: str
    actual_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    class_mismatch_count: int
    finite_mismatch_count: int
    max_abs_error: float | None

    def __bool__(self) -> bool:
        return self.passed


def _validate_tolerance(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def compare_semantic_outputs(
    actual: Any,
    reference: Any,
    *,
    atol: Real,
    rtol: Real,
) -> SemanticComparison:
    """Compare BF16/float outputs by class, then by the supplied tolerance.

    NaN payload and sign bits are intentionally absent from the comparison.
    Infinities must have the same sign.  Finite values use exactly
    ``atol + rtol * abs(reference)`` and shapes must match exactly.
    """

    absolute_tolerance = _validate_tolerance(atol, "atol")
    relative_tolerance = _validate_tolerance(rtol, "rtol")
    actual_values = _output_values(actual, "actual output")
    reference_values = _output_values(reference, "reference output")
    actual_shape = tuple(int(dimension) for dimension in actual_values.shape)
    reference_shape = tuple(int(dimension) for dimension in reference_values.shape)
    if actual_shape != reference_shape:
        return SemanticComparison(
            passed=False,
            reason=f"shape mismatch: actual {actual_shape}, reference {reference_shape}",
            actual_shape=actual_shape,
            reference_shape=reference_shape,
            class_mismatch_count=0,
            finite_mismatch_count=0,
            max_abs_error=None,
        )

    actual_classes = classify_semantic_output(actual_values)
    reference_classes = classify_semantic_output(reference_values)
    class_mismatch = actual_classes != reference_classes
    class_mismatch_count = int(np.count_nonzero(class_mismatch))

    finite_pair = (actual_classes == "finite") & (reference_classes == "finite")
    if np.any(finite_pair):
        # Report the unscaled error in float64, preserving inf when the true
        # subtraction overflows. For the pass/fail decision, compare scaled
        # quantities so neither the difference nor rtol*abs(reference) can
        # overflow before the mathematically equivalent comparison is made.
        finite_actual = np.asarray(actual_values[finite_pair], dtype=np.float64)
        finite_reference = np.asarray(reference_values[finite_pair], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_error = np.abs(
                np.subtract(finite_actual, finite_reference, dtype=np.float64)
            )
            scale = np.maximum(
                np.maximum(np.abs(finite_actual), np.abs(finite_reference)),
                np.float64(absolute_tolerance),
            )
            nonzero_scale = scale != np.float64(0.0)
            scaled_actual = np.zeros_like(finite_actual)
            scaled_reference = np.zeros_like(finite_reference)
            scaled_atol = np.zeros_like(scale)
            np.divide(finite_actual, scale, out=scaled_actual, where=nonzero_scale)
            np.divide(finite_reference, scale, out=scaled_reference, where=nonzero_scale)
            np.divide(
                np.float64(absolute_tolerance),
                scale,
                out=scaled_atol,
                where=nonzero_scale,
            )
            scaled_difference = np.abs(
                np.subtract(scaled_actual, scaled_reference, dtype=np.float64)
            )
            scaled_bound = np.add(
                scaled_atol,
                np.multiply(
                    np.float64(relative_tolerance),
                    np.abs(scaled_reference),
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        finite_fail = scaled_difference > scaled_bound
        finite_mismatch_count = int(np.count_nonzero(finite_fail))
        max_abs_error = float(np.max(absolute_error))
    else:
        finite_mismatch_count = 0
        max_abs_error = None

    passed = class_mismatch_count == 0 and finite_mismatch_count == 0
    if passed:
        reason = "semantic output matches"
    elif class_mismatch_count:
        reason = f"output class mismatch at {class_mismatch_count} element(s)"
    else:
        reason = f"finite tolerance mismatch at {finite_mismatch_count} element(s)"
    return SemanticComparison(
        passed=passed,
        reason=reason,
        actual_shape=actual_shape,
        reference_shape=reference_shape,
        class_mismatch_count=class_mismatch_count,
        finite_mismatch_count=finite_mismatch_count,
        max_abs_error=max_abs_error,
    )


def assert_semantic_outputs_close(
    actual: Any,
    reference: Any,
    *,
    atol: Real,
    rtol: Real,
) -> SemanticComparison:
    """Raise ``AssertionError`` when semantic output comparison fails."""

    comparison = compare_semantic_outputs(actual, reference, atol=atol, rtol=rtol)
    if not comparison:
        raise AssertionError(comparison.reason)
    return comparison


# Explicit aliases keep the storage-vs-value boundary discoverable to callers.
compare_bf16_outputs = compare_semantic_outputs
assert_bf16_outputs_close = assert_semantic_outputs_close
