# Pre-SQ T4 Reference Guard Summary 2026-07-06

## Fixture

```text
benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16
```

## Package

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

## Results

| target | uLLM device | backend | layers | max MSE | max mean abs diff | max abs diff | min cosine similarity | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | hip | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |
| V620/RDNA2 | `1` | hip | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |

## Raw Artifacts

- `benchmarks/results/2026-07-06/engine/package-golden-prefix-t4-r9700-actual-prefix0-12-accepted-qwen35-hidden3994-v1.jsonl`
- `benchmarks/results/2026-07-06/engine/package-golden-prefix-t4-v620-actual-prefix0-12-accepted-qwen35-hidden3994-v1.jsonl`

## Interpretation

This guard does not verify full logits or generated token IDs. It verifies that the accepted package still matches the existing short golden prefix hidden-state fixture across layers `0..12` on both RDNA4 and RDNA2. That is sufficient for the current pre-sq TPS records, whose correctness requirement is to avoid measuring an obviously broken runtime path.
