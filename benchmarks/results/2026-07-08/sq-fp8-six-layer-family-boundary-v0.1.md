# SQ FP8 Six-Layer Family Boundary v0.1

Date: 2026-07-08

## Summary

This guard narrows the 6-layer strict-top1 boundary for the SQ FP8 row-block candidate.

Context:

- Previous mixed policy: `v` fallback + `q/k/o/gate/up/down` row-block32 FP8.
- That policy passed 4-5 layer short guards, but failed layers `3,7,11,15,19,23`.
- Layer `23` alone was fixed by `q/v` fallback, but 6-layer `q/v` fallback still failed.

This guard tests the remaining families over layers `3,7,11,15,19,23`.

Result:

- `k` row-block32 passes strict top1.
- `up` row-block32 passes strict top1.
- `k+up` row-block32 passes strict top1 across `3 / 3` short prompts.
- `o`, `gate`, and `down` row-block32 fail strict top1 individually.
- `o`, `gate`, and `down` row-block16 still fail strict top1 individually.

Therefore the strict-top1-safe 6-layer subset found so far is only:

```text
FP8 row-block32: k/up
fallback: q/v/o/gate/down
```

This is a useful regression boundary, not a final SQ policy. It keeps too little of the model in
FP8 to serve as the intended SQ format candidate.

## Conditions

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Layers: `3,7,11,15,19,23`
- Top-k: `8`
- LM head chunk rows: `4096`

Overlay timing is not a throughput result. The current path materializes selected FP8 tensors for
quality boundary testing.

## Six-Layer Family Split

Scale: row-block32.

| family | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `k` | `6` | `70114` | `70114` | true | `1` | `8 / 8` | `0.000000` |
| `o` | `6` | `70114` | `157558` | false | `3` | `6 / 8` | `0.184640` |
| `gate` | `6` | `70114` | `157558` | false | `2` | `7 / 8` | `0.209001` |
| `up` | `6` | `70114` | `70114` | true | `1` | `7 / 8` | `0.000000` |
| `down` | `6` | `70114` | `157558` | false | `3` | `7 / 8` | `0.108557` |

## Stronger Row-Block Check

Scale: row-block16.

| family | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `o` | `6` | `70114` | `157558` | false | `3` | `6 / 8` | `0.145383` |
| `gate` | `6` | `70114` | `157558` | false | `2` | `7 / 8` | `0.247415` |
| `down` | `6` | `70114` | `157558` | false | `3` | `7 / 8` | `0.074723` |

Row-block16 improves the gap for `o` and `down`, but it does not recover strict top1.

## k/up Combined Guard

Scale: row-block32.

| case | token IDs | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | diagnostic pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| len4 | `1,2,3,4` | `12` | `70114` | `70114` | true | `1` | `7 / 8` | true |
| case_a | `100,200,300,400,500,600,700,800` | `12` | `192493` | `192493` | true | `1` | `2 / 8` | false |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | `12` | `246866` | `246866` | true | `1` | `6 / 8` | true |

Acceptance summary from `sq-fp8-layers3-7-11-15-19-23-k-up-acceptance-v0.1.json`:

| field | value |
| --- | ---: |
| case count | `3` |
| strict top1 pass count | `3` |
| strict top1 passed | `true` |
| diagnostic top-k pass count | `2` |
| diagnostic top-k passed | `false` |
| accepted by strict-top1 evaluator | `true` |

The acceptance result applies only to this partial `k/up` overlay target set. It does not mean the
full SQ FP8 candidate is accepted.

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-family-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-ogatedown-rowblock16-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-k-up-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-k-up-rowblock32-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-k-up-rowblock32-case-b-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-k-up-acceptance-v0.1.json`

## Next Action

1. Keep `k/up` row-block32 as the current 6-layer strict-top1 regression subset.
2. Do not promote it as the SQ policy because coverage is too low.
3. Continue T2 by testing stronger formats or fallback policies for `q/v/o/gate/down`.
4. Prioritize `o/down` before `gate` if using diagnostic gap as the ordering signal; row-block16
   moved `o/down` closer but not enough.
