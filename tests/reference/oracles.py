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
