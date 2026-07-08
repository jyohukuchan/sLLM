# SQ FP8 Mixed Candidate Layer Scaling Guard v0.1

## Summary

This guard expands the row-block mixed SQ FP8 candidate found in
`sq-fp8-rowblock-scale-risk-guard-v0.1.md`.

Candidate:

- FP8 row-block32: `q/k/o/gate/up/down`
- fallback to AQ4 package tensors: `v`
- scale dtype: `f32`
- payload dtype: `fp8_e4m3`

The mixed candidate is useful, but it is not yet a full-target SQ policy.

- Layers `3,7,11,15` pass strict top1 on `3 / 3` short prompt cases.
- Layers `3,7,11,15,19` pass strict top1 on the len4 case.
- Layers `3,7,11,15,19,23` fail strict top1 on the len4 case.
- All self-attention probe layers `3,7,11,15,19,23,27,31` fail strict top1.
- Layer `23` is a boundary layer. `q` alone fails under row-block32; `q/v` fallback fixes layer `23` alone, but not the 6-layer bundle.

The immediate conclusion is that row-block32 plus `v` fallback is a partial T2 quality policy,
not the final SQ format candidate. Full-target promotion still needs an acceptance rule and either
more fallback, a stronger format for sensitive families/layers, or text-level guard evidence.

## Conditions

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Top-k: `8`
- LM head chunk rows: `4096`
- Scale granularity: `row_block`
- Scale block cols: `32`

The layer load timings in these rows are not SQ throughput results. The current guard path
materializes selected FP8 tensors through the overlay path and is intended for quality boundary
testing only.

## Mixed Candidate Scaling

| run | layers | token IDs | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 4layer_case_a | `3,7,11,15` | `100,200,300,400,500,600,700,800` | `24` | `197397` | `197397` | true | `1` | `4 / 8` | `0.000000` |
| 4layer_len4 | `3,7,11,15` | `1,2,3,4` | `24` | `187677` | `187677` | true | `1` | `5 / 8` | `0.000000` |
| 4layer_case_b | `3,7,11,15` | `42,314,2718,1618,12345,23456,34567,45678` | `24` | `219331` | `219331` | true | `1` | `5 / 8` | `0.000000` |
| 5layer_len4 | `3,7,11,15,19` | `1,2,3,4` | `30` | `120026` | `120026` | true | `1` | `6 / 8` | `0.000000` |
| 6layer_len4 | `3,7,11,15,19,23` | `1,2,3,4` | `36` | `70114` | `157558` | false | `5` | `6 / 8` | `0.250585` |
| all_self_attn_len4 | `3,7,11,15,19,23,27,31` | `1,2,3,4` | `48` | `140864` | `222957` | false | not in top8 | `4 / 8` | n/a |

Interpretation:

- The previous `4layer_case_a` failure for the row-scale safe subset is recovered by the mixed
  row-block32 candidate.
- The first observed len4 expansion failure is at layer `23`.
- The all-self-attention probe confirms that the policy cannot be promoted by simply expanding
  it to more layers.

## Layer 23 Boundary

| run | layers | fallback | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| layer23_except_v | `23` | `v` | `6` | `108449` | `29487` | false | `2` | `5 / 8` | `0.014240` |
| layer23_except_qv | `23` | `q/v` | `5` | `108449` | `108449` | true | `1` | `6 / 8` | `0.000000` |
| 6layer_except_v_q23 | `3,7,11,15,19,23` | `v` all layers, `q` only layer 23 | `35` | `70114` | `157558` | false | `5` | `6 / 8` | `0.214042` |
| 6layer_except_qv | `3,7,11,15,19,23` | `q/v` all layers | `30` | `70114` | `157558` | false | `4` | `5 / 8` | `0.124241` |

Layer `23` alone is fixed by `q/v` fallback, but the 6-layer bundle still fails even when `q/v`
fall back in all six layers. That points to cumulative drift in the remaining
`k/o/gate/up/down` row-block32 tensors, not just a single `q` tensor in layer `23`.

## Layer 23 Family Split

| family | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `q` | `1` | `108449` | `139555` | false | `2` | `7 / 8` | `0.061239` |
| `k` | `1` | `108449` | `108449` | true | `1` | `7 / 8` | `0.000000` |
| `o` | `1` | `108449` | `108449` | true | `1` | `5 / 8` | `0.000000` |
| `gate` | `1` | `108449` | `108449` | true | `1` | `6 / 8` | `0.000000` |
| `up` | `1` | `108449` | `108449` | true | `1` | `8 / 8` | `0.000000` |
| `down` | `1` | `108449` | `108449` | true | `1` | `7 / 8` | `0.000000` |

## Storage Snapshot

| artifact | FP8 tensors | passthrough tensors | FP8 payload bytes | FP8 scale bytes | compact resident bytes estimate | materialized working-set bytes estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4layer mixed candidate | `24` | `751` | `822083584` | `102760448` | `18586893280` | `201326592` |
| all self-attention mixed candidate | `48` | `727` | `1644167168` | `205520896` | `17867570144` | `201326592` |

The compact resident estimate still includes passthrough source bytes and is conservative for the
v0.1 artifact boundary.

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-except-v-rowblock32-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-except-v-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-except-v-rowblock32-case-b-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-except-v-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-except-v-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-all-self-attn-except-v-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layer23-except-v-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layer23-family-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-except-v-q23-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layer23-except-qv-rowblock32-len4-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-11-15-19-23-except-qv-rowblock32-len4-v0.1.json`

## Next Action

1. Keep the mixed row-block32 policy as the current partial-quality candidate, not the final SQ
   policy.
2. Define the T2 short guard acceptance rule before promoting broader SQ coverage.
3. Investigate 6-layer cumulative drift with one of:
   - additional fallback families for deep layers;
   - per-layer sensitivity policy;
   - stronger scale/layout for sensitive tensors;
   - text-level guard if strict top1 is too brittle.
4. Do not treat the overlay load timing as SQ throughput. Throughput comparison still needs the
   real batch runner and native/materialization-aware runtime path.
