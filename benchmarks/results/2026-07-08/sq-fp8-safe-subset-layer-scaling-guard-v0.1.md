# SQ FP8 Safe Subset Layer Scaling Guard v0.1

## Summary

This guard expands the strict-top1-safe subset from layers `3,7` to larger self-attention layer
sets.

Safe subset:

- self-attention `k_proj`
- self-attention `o_proj`
- MLP `gate_proj`
- MLP `up_proj`

Risky families `q_proj`, `v_proj`, and `down_proj` are intentionally excluded.

The purpose is to check whether the safe subset from the layers `3,7` family split remains stable
when more layers are affected.

## Results

| run | layers | token IDs | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2layer_len4 | `3,7` | `1,2,3,4` | `8` | `33604` | `33604` | true | `1` | `7 / 8` | `0.000000` |
| 2layer_case_a | `3,7` | `100,200,300,400,500,600,700,800` | `8` | `15611` | `15611` | true | `1` | `5 / 8` | `0.000000` |
| 2layer_case_b | `3,7` | `42,314,2718,1618,12345,23456,34567,45678` | `8` | `227701` | `227701` | true | `1` | `7 / 8` | `0.000000` |
| 3layer_case_a_3_7_11 | `3,7,11` | `100,200,300,400,500,600,700,800` | `12` | `184137` | `184137` | true | `1` | `6 / 8` | `0.000000` |
| 1layer_case_a_15 | `15` | `100,200,300,400,500,600,700,800` | `4` | `402` | `402` | true | `1` | `6 / 8` | `0.000000` |
| 3layer_case_a_3_7_15 | `3,7,15` | `100,200,300,400,500,600,700,800` | `12` | `67893` | `67893` | true | `1` | `3 / 8` | `0.000000` |
| 4layer_len4 | `3,7,11,15` | `1,2,3,4` | `16` | `187677` | `187677` | true | `1` | `6 / 8` | `0.000000` |
| 4layer_case_a | `3,7,11,15` | `100,200,300,400,500,600,700,800` | `16` | `197397` | `244381` | false | `3` | `4 / 8` | `0.137706` |
| 4layer_case_b | `3,7,11,15` | `42,314,2718,1618,12345,23456,34567,45678` | `16` | `219331` | `219331` | true | `1` | `5 / 8` | `0.000000` |

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-policy-subset-guard-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-safe-subset-case-b-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layer15-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-15-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-safe-subset-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-safe-subset-case-b-v0.1.json`

## Interpretation

The safe subset is not globally safe under strict top1.

- Layers `3,7` passed `3 / 3` short prompts.
- Layer set `3,7,11` passed the previously failing `case_a`.
- Layer `15` alone passed `case_a`.
- Layer set `3,7,15` passed `case_a`.
- Layer set `3,7,11,15` passed `2 / 3` short prompts but failed `case_a`.

The `4layer_case_a` failure is cumulative or interaction-driven rather than a simple layer 15
single-layer failure. Under strict top1, `k/o/gate/up` can be used as the first expansion subset,
but it still needs layer-count and prompt-bundle qualification before being treated as a full SQ
candidate target.

The timing includes host-side FP8 to F32 materialization and should not be read as native FP8
throughput.

## Next Action

1. Keep strict-top1 qualification for any expanded SQ target set.
2. Treat `k/o/gate/up` as a candidate subset, not a proven-safe policy.
3. Use the failing `4layer_case_a` as a regression guard when changing scale policy.
4. Test `q/v/down` with a stronger scale or fallback policy before trying a full-target SQ artifact.
