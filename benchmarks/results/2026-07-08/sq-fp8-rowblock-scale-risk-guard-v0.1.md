# SQ FP8 Row-Block Scale Risk Guard v0.1

## Summary

This guard adds and evaluates `row_block` scale for the risky SQ FP8 families identified in the
layers `3,7` family split.

`row_block` keeps the FP8 payload at 1 byte per weight value, but stores F32 scales for each
`row x column-block` instead of one scale per row.

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Layers: `3,7`
- Main token IDs: `1,2,3,4`
- Top-k: `8`

## Implementation

- `tools/build-sq-fp8-w8a16-artifact.py` now accepts `--scale-granularity row_block`.
- `tools/build-sq-fp8-w8a16-artifact.py` now accepts `--scale-block-cols N`.
- `sq_manifest.json` tensor entries may include `scale_block_cols`.
- `crates/ullm-engine/src/sq.rs` can validate and materialize `row_block` scaled tensors.
- `tools/run-sq-fp8-overlay-logits-guard.py` can pass scale options to the artifact builder.

## Risk Subset Results

Baseline AQ4 top8 for `layers=3,7`, token IDs `1,2,3,4`:

`33604,239469,103346,49290,80054,35188,239762,148443`

| case | scale | block cols | FP8 tensors | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `q/v/down` | row | n/a | `6` | `239469` | false | `5` | `6 / 8` | `0.406537` |
| `q/v/down` | row_block | `256` | `6` | `239469` | false | `5` | `6 / 8` | `0.368687` |
| `q/v/down` | row_block | `64` | `6` | `239469` | false | `5` | `6 / 8` | `0.287833` |
| `q/v/down` | row_block | `16` | `6` | `239469` | false | `6` | `6 / 8` | `0.384809` |

Row-block scale improves the risky group at block `64`, but does not recover strict top1 when all
three risky families are FP8.

## Family Split With Row-Block Scale

| case | scale | block cols | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `q` | row_block | `64` | `239469` | false | `3` | `7 / 8` | `0.019917` |
| `q` | row_block | `32` | `33604` | true | `1` | `7 / 8` | `0.000000` |
| `v` | row_block | `128` | `239469` | false | `2` | `7 / 8` | `0.152845` |
| `v` | row_block | `64` | `239469` | false | `2` | `7 / 8` | `0.079760` |
| `v` | row_block | `32` | `239469` | false | `2` | `7 / 8` | `0.064440` |
| `v` | row_block | `16` | `239469` | false | `2` | `7 / 8` | `0.120411` |
| `down` | row_block | `64` | `33604` | true | `1` | `7 / 8` | `0.000000` |

Observations:

- `q` can recover strict top1 at block `32`.
- `down` can recover strict top1 at block `64`.
- `v` does not recover strict top1 for the tested block sizes.
- For `v`, block `32` gives the smallest measured top1 gap in this guard, but still leaves AQ4 top1
  at rank `2`.

## Mixed Candidate Check

Candidate:

- FP8 row-block32: `q/k/o/gate/up/down`
- fallback: `v`

| case | token IDs | FP8 tensors | AQ4 top1 | SQ top1 | top1 match | top8 common |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| len4 | `1,2,3,4` | `12` | `33604` | `33604` | true | `6 / 8` |
| case_a | `100,200,300,400,500,600,700,800` | `12` | `15611` | `15611` | true | `3 / 8` |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | `12` | `227701` | `227701` | true | `5 / 8` |

Mixed candidate top1 match count: `3 / 3`.

Artifact storage for len4 mixed candidate:

| field | value |
| --- | ---: |
| FP8 tensor count | `12` |
| FP8 payload bytes | `411041792` |
| FP8 scale bytes | `51380224` |
| compact resident bytes estimate | `18946554848` |
| scale block cols | `32` |

## Raw JSON

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-risk-rowblock256-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-risk-rowblock64-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-risk-rowblock16-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-risk-family-rowblock64-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-qv-rowblock32-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-v-rowblock16-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-v-rowblock128-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-except-v-rowblock32-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-except-v-rowblock32-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-except-v-rowblock32-case-b-v0.1.json`

## Interpretation

`row_block` is useful but not sufficient for a full FP8 target under strict top1.

The most promising near-term policy from this guard is:

1. keep `v_proj` as fallback;
2. use row-block32 FP8 for `q/k/o/gate/up/down`;
3. keep `4layer_case_a` and the layers `3,7` mixed-candidate bundle as regression guards.

This is still not a final SQ policy. It is a T2 short-quality boundary that narrows the next
candidate from "all target projections FP8" to "row-block FP8 except `v_proj` fallback" for the
strict-top1 path.
