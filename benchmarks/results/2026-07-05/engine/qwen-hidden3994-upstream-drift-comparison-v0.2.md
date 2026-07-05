# Qwen Hidden3994 Upstream Drift Comparison v0.2

## Key Findings

- tokens1 layer8 baseline output_diff is positive, while tokens401 layer8 baseline output_diff is negative and is the final worst coordinate.
- layer8 up6340 scale 1.008 increases tokens1 final max abs by 0.000799179077, still inside the 0.001 hard gate.
- the same candidate improves tokens201 by 0.000621795654 but only changes tokens401 by +0.000033378601, so it does not address the tokens401 input-drift amplification.

## Layer8 and Final Coordinates

| fixture | layer | baseline input | baseline delta | baseline output | candidate output | output delta | baseline max | candidate max | max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 8 | -0.156290054 | 0.452468872 | 0.296178818 | 0.294593811 | -0.00158500671 | 0.578010559 | 0.57925415 | 0.00124359131 |
| tokens1 | 11 | -0.376991272 | -0.268346786 | -0.645338058 | -0.646137238 | -0.000799179077 | 0.645338058 | 0.646137238 | 0.000799179077 |
| tokens401 | 8 | -0.464719772 | -0.494586945 | -0.959306717 | -0.959340096 | -3.33786011e-05 | 0.959306717 | 0.959340096 | 3.33786011e-05 |
| tokens401 | 11 | -0.135557175 | 0.407335281 | 0.271778107 | 0.271751404 | -2.67028809e-05 | 0.813562393 | 0.813598633 | 3.6239624e-05 |

## Source Chains

- `tokens1-baseline`: `benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-tokens1-baseline-token7-hidden3994-v0.2.json`
- `tokens1-layer8-up6340-s1p008`: `benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-tokens1-layer8-up6340-s1p008-token7-hidden3994-v0.2.json`
- `tokens401-baseline`: `benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-tokens401-baseline-token9-hidden3994-v0.2.json`
- `tokens401-layer8-up6340-s1p008`: `benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-tokens401-layer8-up6340-s1p008-token9-hidden3994-v0.2.json`
