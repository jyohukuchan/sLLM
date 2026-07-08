# SQ FP8 Six-Layer Per-Layer Combination Boundary v0.1

Date: 2026-07-08

## Summary

This guard continues the 6-layer SQ FP8 row-block32 boundary search.

Previous result:

- Over layers `3,7,11,15,19,23`, `k/up` row-block32 passed `3 / 3` strict-top1 short prompts.
- Over the same 6 layers, `o/gate/down` row-block32 failed strict top1 individually.
- `o/gate/down` row-block16 also failed strict top1 individually.

New result:

- Over the first 5 tested layers `3,7,11,15,19`, `o/gate/down` row-block32 each pass strict top1.
- With `k/up` FP8 on all 6 layers, adding any one of `o5`, `gate5`, or `down5` passes len4.
- With `k/up` FP8 on all 6 layers, adding any two of `o5`, `gate5`, and `down5` passes len4.
- Adding all three of `o5/gate5/down5` with `k/up` all 6 layers fails len4.

This means the current len4 frontier is:

```text
FP8 row-block32:
  k/up on layers 3,7,11,15,19,23
  any two of o/gate/down on layers 3,7,11,15,19

fallback:
  q/v
  layer 23 o/gate/down
  one of o/gate/down on layers 3,7,11,15,19
```

This is still not a promoted SQ policy. It is a T2 quality boundary for choosing the next prompt
bundle candidate.

## Conditions

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Top-k: `8`
- LM head chunk rows: `4096`
- Scale: row-block32
- Main token IDs: `1,2,3,4`

Overlay timing is not a throughput result.

## Five-Layer Family Check

Layers: `3,7,11,15,19`.

| family | FP8 tensors | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | --- | ---: | ---: | ---: |
| `o` | `5` | true | `1` | `8 / 8` | `0.000000` |
| `gate` | `5` | true | `1` | `7 / 8` | `0.000000` |
| `down` | `5` | true | `1` | `7 / 8` | `0.000000` |

## k/up Plus One Family

Layers:

- `k/up`: `3,7,11,15,19,23`
- extra family: `3,7,11,15,19`

| case | FP8 tensors | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | --- | ---: | ---: | ---: |
| `kup6_o5` | `17` | true | `1` | `6 / 8` | `0.000000` |
| `kup6_gate5` | `17` | true | `1` | `7 / 8` | `0.000000` |
| `kup6_down5` | `17` | true | `1` | `7 / 8` | `0.000000` |

## k/up Plus Two Families

Layers:

- `k/up`: `3,7,11,15,19,23`
- extra families: `3,7,11,15,19`

| case | FP8 tensors | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | --- | ---: | ---: | ---: |
| `kup6_o5_gate5` | `22` | true | `1` | `6 / 8` | `0.000000` |
| `kup6_o5_down5` | `22` | true | `1` | `5 / 8` | `0.000000` |
| `kup6_gate5_down5` | `22` | true | `1` | `7 / 8` | `0.000000` |

## k/up Plus Three Families

Layers:

- `k/up`: `3,7,11,15,19,23`
- `o/gate/down`: `3,7,11,15,19`

| case | FP8 tensors | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | --- | ---: | ---: | ---: |
| `kup6_ogatedown5` | `27` | false | `3` | `5 / 8` | `0.031211` |

The three-family candidate is close by logit gap, but it fails the strict-top1 T2 promotion rule.

## Interpretation

The failure is not caused by any single one of `o5`, `gate5`, or `down5` when added to `k/up`.
It appears when all three are combined.

For the next prompt-bundle check, the strongest len4 candidate by top-k overlap is:

```text
kup6_gate5_down5
```

It has `22` FP8 tensors, strict top1 match, AQ4 top1 rank `1`, and `7 / 8` top8 common on len4.

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-ogatedown-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup-plus-one-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup-plus-two-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-six-layer-per-layer-candidate-rowblock32-len4-v0.1.json`

## Next Action

1. Keep `kup6_gate5_down5` as the next 6-layer prompt-bundle candidate.
2. Run it on case_a and case_b before treating it as a regression subset.
3. Keep `kup6_ogatedown5` as a strict-top1 failure case and near-miss diagnostic.
4. Do not promote any of these to full SQ policy until prompt-bundle qualification and broader
   coverage are satisfied.
