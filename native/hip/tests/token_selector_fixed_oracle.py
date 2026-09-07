#!/usr/bin/env python3
"""Independent NumPy oracle fixture for the fixed token selector ABI.

The GPU test can use the emitted token/logprob pair as a review fixture.  The
calculation intentionally performs BF16 truncation before stable score sorting,
then applies top-k, inclusive top-p, and the shared SplitMix64 counter draw.
"""

import json
import math

import numpy as np


def splitmix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def main() -> None:
    logits = np.array(
        [3.0, 3.0, 2.5, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5,
         -1.0, -1.0, -1.5, -2.0, -2.0, -2.5, -3.0, -3.0, -3.0,
         -3.5, -4.0, -4.5, -5.0, -5.5, -6.0], dtype=np.float32)
    # Match float_to_bf16 in the C ABI test: truncation, rather than rounding.
    logits = (logits.view(np.uint32) >> 16).astype(np.uint16)
    effective = np.array(
        [0.0, 0.0, 0.1, -0.1, 0.05, 0.2, 0.0, 0.15, 0.0, 0.1,
         0.0, 0.05, 0.0, 0.1, 0.0, 0.05, 0.0, 0.1, 0.0, 0.05,
         0.0, 0.1, 0.0, 0.05, 0.0], dtype=np.float32)
    mask = np.array([(index % 7) != 1 and (index % 11) != 4
                     for index in range(logits.size)], dtype=bool)
    values = (logits.astype(np.uint32) << 16).view(np.float32) + effective
    order = sorted(np.flatnonzero(mask), key=lambda index: (-float(values[index]), int(index)))[:20]
    weights = np.exp(values[order] - values[order[0]])
    cumulative = np.cumsum(weights)
    included = int(np.searchsorted(cumulative, 0.95 * weights.sum(), side="left")) + 1
    seed = 0x123456789ABCDEF0
    counter = 3
    draw = splitmix64((seed + (counter + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    target = ((draw >> 11) / float(1 << 53)) * cumulative[included - 1]
    selected = order[included - 1]
    for index, mass in enumerate(np.cumsum(weights[:included])):
        if target < mass:
            selected = order[index]
            break
    result = {
        "vocab": int(logits.size),
        "top_k": 20,
        "top_p": 0.95,
        "seed": seed,
        "counter": counter,
        "mask": mask.astype(np.uint8).tolist(),
        "logits_bf16": [int(value) for value in logits],
        "additive": effective.tolist(),
        "expected_token_id": int(selected),
        "expected_logprob": float(math.log(float(weights[order.index(selected)] / cumulative[included - 1]))),
        # Independent all-tied V=100 boundary sequence used by the GPU test.
        "tie_vocab100_expected_tokens": [
            int(((splitmix64((seed_value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) >> 11)
                 / float(1 << 53)) * 95.0)
            for seed_value in range(64)
        ],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
