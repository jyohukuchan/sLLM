# SQ FP8 Six-Layer `kup6_gate5_down5` Prompt Bundle v0.1

Date: 2026-07-08

## Summary

This guard expands the len4 `kup6_gate5_down5` candidate to the short prompt bundle.

Result:

- `kup6_gate5_down5` passes strict top1 on len4, case_a, and case_b.
- The candidate has `22` FP8 row-block32 tensors.
- It is a stronger six-layer regression subset than `k/up` alone.
- It is still not a promoted full SQ policy because case_a has low top8 overlap and coverage is limited.

Current subset:

```text
FP8 row-block32:
  k/up on layers 3,7,11,15,19,23
  gate/down on layers 3,7,11,15,19

fallback:
  q/v/o
  gate/down on layer 23
```

## Conditions

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Top-k: `8`
- LM head chunk rows: `4096`
- Scale: row-block32
- Layers: `3,7,11,15,19,23`

Overlay timing is not a throughput result.

## Prompt Bundle

| case | token IDs | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| len4 | `1,2,3,4` | `22` | `70114` | `70114` | true | `1` | `7 / 8` | `0.000000` |
| case_a | `100,200,300,400,500,600,700,800` | `22` | `192493` | `192493` | true | `1` | `2 / 8` | `0.000000` |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | `22` | `246866` | `246866` | true | `1` | `6 / 8` | `0.000000` |

## Interpretation

`kup6_gate5_down5` is the current best six-layer strict-top1 subset.

The low `2 / 8` top8 overlap on case_a means it should not be treated as a final quality policy.
It is suitable as a regression subset and as a candidate boundary for the next SQ artifact policy.

Compared with `kup6_ogatedown5`, which fails len4 strict top1, this subset is the better next
candidate because it keeps all three prompt-bundle top1s stable while preserving more FP8 tensors
than `k/up` alone.

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup-plus-two-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup6-gate5-down5-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup6-gate5-down5-case-b-v0.1.json`

## Next Action

1. Keep `kup6_gate5_down5` as the current six-layer strict-top1 regression subset.
2. Do not promote it as the full SQ policy until broader coverage or an accepted text-level guard exists.
3. Move T2 from boundary search toward manifest/policy representation for selected FP8 and fallback families.
4. Continue T1 real batch runner work before using throughput rows for SQ performance decisions.
