"""Independent, bounded NumPy references for H2.

These helpers intentionally operate on metadata and tiny vectors only. They do
not reproduce a production kernel or allocate a model-shaped tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np


DEFAULT_SEED = 20260803
MAX_CASES = 8
MAX_ORACLE_ELEMENTS = 4096
RMSNORM_OFFSET_ONE = "offset_one"


def bf16_encode_rne(values: Any) -> np.ndarray:
    """Encode float32 values to BF16 bit patterns using round-to-nearest-even.

    NumPy has no portable BF16 scalar dtype, so the oracle represents BF16
    values as uint16 storage bits. NaN payloads are kept nonzero rather than
    being allowed to round into infinity.
    """

    float_values = np.asarray(values, dtype=np.float32)
    bits = float_values.view(np.uint32)
    round_bits = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + round_bits) >> np.uint32(16)).astype(np.uint16)

    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    nan_mask = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    preserved_nan = (bits >> np.uint32(16)).astype(np.uint16) | np.uint16(1)
    return np.where(nan_mask, preserved_nan, rounded).astype(np.uint16, copy=False)


def bf16_decode(codes: Any) -> np.ndarray:
    """Decode uint16 BF16 storage bits to float32 without changing bit shape."""

    raw_codes = np.asarray(codes)
    if raw_codes.dtype != np.uint16:
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise ValueError("BF16 codes must be an unsigned 16-bit integer array")
        if np.any(raw_codes < 0) or np.any(raw_codes > np.iinfo(np.uint16).max):
            raise ValueError("BF16 codes must fit in uint16")
        raw_codes = raw_codes.astype(np.uint16)
    bits = raw_codes.astype(np.uint32, copy=False) << np.uint32(16)
    return bits.view(np.float32)


# Short aliases make the representation boundary explicit at call sites.
encode_bf16_rne = bf16_encode_rne
decode_bf16 = bf16_decode


def _bf16_input_codes(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == np.uint16:
        codes = raw
    elif np.issubdtype(raw.dtype, np.floating):
        values = np.asarray(raw, dtype=np.float32)
        codes = bf16_encode_rne(values)
    else:
        raise ValueError(f"{name} must be BF16 uint16 bits or floating-point values")
    return codes


def rmsnorm_bf16(
    activation: Any,
    raw_scale: Any,
    epsilon: Real,
    scale_mode: str,
) -> np.ndarray:
    """Return BF16-bit RMSNorm output for a bounded tiny case.

    Inputs supplied as floating-point arrays are first rounded to BF16; uint16
    inputs are treated as already-encoded BF16 storage. Reduction, epsilon,
    inverse RMS, and the explicit offset-one scale are all evaluated in
    float32. Activation and raw-scale nonfinite values are valid evidence and
    propagate through the IEEE float32 calculation. There is intentionally no
    default for ``epsilon`` or ``scale_mode``.
    """

    if scale_mode != RMSNORM_OFFSET_ONE:
        raise ValueError("RMSNorm scale_mode must be the explicit offset_one mode")
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError("RMSNorm epsilon must be finite and positive")
    try:
        epsilon_value = np.float32(epsilon)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("RMSNorm epsilon must be finite and positive") from exc
    if not np.isfinite(epsilon_value) or epsilon_value <= np.float32(0.0):
        raise ValueError("RMSNorm epsilon must be finite and positive")

    activation_codes = _bf16_input_codes(activation, "activation")
    scale_codes = _bf16_input_codes(raw_scale, "raw scale")
    if activation_codes.ndim == 0:
        raise ValueError("RMSNorm activation must have rank at least one")
    if activation_codes.size == 0 or scale_codes.size == 0:
        raise ValueError("RMSNorm activation and raw scale must be non-empty")
    if activation_codes.size > MAX_ORACLE_ELEMENTS or scale_codes.size > MAX_ORACLE_ELEMENTS:
        raise ValueError("RMSNorm evidence exceeds the tiny oracle element bound")
    if scale_codes.ndim != 1:
        raise ValueError("RMSNorm raw scale must have rank one")
    if activation_codes.shape[-1] != scale_codes.shape[0]:
        raise ValueError("RMSNorm raw scale length must match the activation last dimension")

    x = bf16_decode(activation_codes).astype(np.float32, copy=False)
    raw = bf16_decode(scale_codes).astype(np.float32, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        squared = np.multiply(x, x, dtype=np.float32)
        sum_squares = np.sum(squared, axis=-1, keepdims=True, dtype=np.float32)
        mean_square = np.divide(
            sum_squares,
            np.float32(activation_codes.shape[-1]),
            dtype=np.float32,
        )
        denominator = np.add(mean_square, epsilon_value, dtype=np.float32)
        inverse_rms = np.reciprocal(np.sqrt(denominator, dtype=np.float32), dtype=np.float32)
        effective_scale = np.add(np.float32(1.0), raw, dtype=np.float32)
        normalized = np.multiply(
            np.multiply(x, inverse_rms, dtype=np.float32),
            effective_scale,
            dtype=np.float32,
        )
    return bf16_encode_rne(normalized)


def rmsnorm_bf16_values(
    activation: Any,
    raw_scale: Any,
    epsilon: Real,
    scale_mode: str,
) -> np.ndarray:
    """Return the same oracle output decoded to float32 for numeric checks."""

    return bf16_decode(rmsnorm_bf16(activation, raw_scale, epsilon, scale_mode))


@dataclass(frozen=True)
class BoundaryCase:
    case_id: str
    values: tuple[int, ...]


def boundary_cases(seed: int = DEFAULT_SEED) -> tuple[BoundaryCase, ...]:
    """Return a deterministic, shuffled set of no-more-than-eight cases."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    cases = (
        BoundaryCase("empty", (0,)),
        BoundaryCase("one", (1,)),
        BoundaryCase("three", (3,)),
        BoundaryCase("seven", (7,)),
        BoundaryCase("sixteen_low_high", (15, 16, 17)),
        BoundaryCase("prime_nonaligned", (17, 37, 73)),
        BoundaryCase("two_fifty_six_low", (255,)),
        BoundaryCase("two_fifty_six_high", (256, 257)),
    )
    order = np.random.default_rng(int(seed)).permutation(len(cases))
    return tuple(cases[int(index)] for index in order)


@dataclass(frozen=True)
class KVLayout:
    layers: int
    kv_planes: int
    batch_size: int
    max_tokens: int
    block_size: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KVLayout":
        fields = {
            "layers",
            "kv_planes",
            "batch_size",
            "max_tokens",
            "block_size",
            "kv_heads",
            "head_dim",
            "dtype_bytes",
        }
        if set(value) != fields:
            raise ValueError("KV layout fields must match the registered contract")
        converted: dict[str, int] = {}
        for field in fields:
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, Integral) or item <= 0:
                raise ValueError(f"{field} must be a positive integer")
            converted[field] = int(item)
        if converted["kv_planes"] != 2:
            raise ValueError("the reference layout has separate K and V planes")
        return cls(**converted)

    @property
    def blocks_per_sequence(self) -> int:
        return (self.max_tokens + self.block_size - 1) // self.block_size

    def byte_offset(
        self,
        *,
        layer: int,
        kv: int,
        batch: int,
        token: int,
        head: int,
        channel: int,
    ) -> int:
        coordinates = {
            "layer": (layer, self.layers),
            "kv": (kv, self.kv_planes),
            "batch": (batch, self.batch_size),
            "token": (token, self.max_tokens),
            "head": (head, self.kv_heads),
            "channel": (channel, self.head_dim),
        }
        for name, (coordinate, bound) in coordinates.items():
            if isinstance(coordinate, bool) or not isinstance(coordinate, Integral):
                raise ValueError(f"{name} must be an integer")
            if not 0 <= coordinate < bound:
                raise ValueError(f"{name} is out of bounds")
        block, in_block = divmod(int(token), self.block_size)
        element = int(layer) * self.kv_planes + int(kv)
        element = element * self.batch_size + int(batch)
        element = element * self.blocks_per_sequence + block
        element = element * self.block_size + in_block
        element = element * self.kv_heads + int(head)
        element = element * self.head_dim + int(channel)
        return element * self.dtype_bytes


def sample_token(
    logits: Any,
    *,
    temperature: float,
    top_p: float,
    seed: int,
) -> int:
    """Sample one token from a bounded vector with stable ordering."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a finite positive number")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    if isinstance(top_p, bool) or not isinstance(top_p, Real):
        raise ValueError("top_p must be in (0, 1]")
    if not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    vector = np.asarray(logits, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or vector.size > MAX_ORACLE_ELEMENTS:
        raise ValueError("logits must be a non-empty tiny vector")
    if not np.isfinite(vector).all():
        raise ValueError("logits must be finite")

    scaled = vector / float(temperature)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities)
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order])
    count = int(np.searchsorted(cumulative, float(top_p), side="left")) + 1
    candidates = order[:count]
    candidate_probabilities = probabilities[candidates]
    candidate_probabilities /= np.sum(candidate_probabilities)
    generator = np.random.default_rng(int(seed))
    return int(generator.choice(candidates, p=candidate_probabilities))
