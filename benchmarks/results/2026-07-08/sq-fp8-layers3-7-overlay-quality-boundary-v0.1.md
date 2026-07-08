# SQ FP8 Layers 3 and 7 Overlay Quality Boundary v0.1

## Summary

This guard expands the SQ FP8 overlay from one self-attention layer to two self-attention layers:
layers `3` and `7`.

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Device: R9700/RDNA4, runtime device index `2`
- Layers: `3,7`
- Token IDs: `1,2,3,4`
- Top-k: `8`
- LM head: AQ4 resident, group size `8`

This is not a real full-model layer order. It is a short guard for checking whether the SQ overlay
path remains quality-stable when more than one self-attention layer is affected.

## Artifacts

| artifact | path | FP8 tensor count | coverage |
| --- | --- | ---: | --- |
| attention-only | `/tmp/ullm-sq-fp8-layers3-7-attn-smoke` | `8` | layers 3/7 `q/k/v/o_proj` |
| MLP-only | `/tmp/ullm-sq-fp8-layers3-7-mlp-smoke` | `6` | layers 3/7 `gate/up/down_proj` |
| attention + MLP | `/tmp/ullm-sq-fp8-layers3-7-projections-smoke` | `14` | layers 3/7 projection set |

## Result

| run | top1 | top1 matches AQ4 | top8 common with AQ4 | AQ4 top1 logit in run | run top1 minus AQ4 top1 logit | layer load ms | total ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| AQ4 baseline | `33604` | true | `8 / 8` | `4.655323` | `0.000000` | `515.786711` | `1346.398054` |
| SQ attention-only | `239469` | false | `6 / 8` | `4.537843` | `0.241863` | `11594.053886` | `12375.153077` |
| SQ MLP-only | `103346` | false | `7 / 8` | `4.571534` | `0.068835` | `27726.445481` | `28495.305660` |
| SQ attention + MLP | `239762` | false | `6 / 8` | `4.462107` | `0.308805` | `35791.201341` | `36528.790309` |

Top8 token IDs:

| run | top8 token IDs |
| --- | --- |
| AQ4 baseline | `33604,239469,103346,49290,80054,35188,239762,148443` |
| SQ attention-only | `239469,239762,33604,49290,146557,103346,246054,35188` |
| SQ MLP-only | `103346,33604,35188,239762,239469,80054,226053,49290` |
| SQ attention + MLP | `239762,239469,103346,35188,33604,146557,148443,246054` |

## Interpretation

This is the first observed T2 quality boundary.

- Layer 3 alone matched top1 for `3 / 3` short cases.
- Layer 7 alone also matched top1 for the `1,2,3,4` case.
- Combining layers `3,7` changes top1 for attention-only, MLP-only, and attention+MLP overlays.
- The AQ4 top1 remains inside SQ top8 for all runs, so this is a ranking drift rather than a
catastrophic output collapse.
- Both attention and MLP projection FP8 overlays can contribute to the drift over multiple layers.

The timing is still not a native FP8 speed result. It includes host-side FP8 to F32 materialization
and runtime F32 copies.

## Next Action

Before promoting `sq-fp8-w8a16-r9700-v0` to full-target T2 guard, split the quality guard by:

1. tensor family: q/k/v/o, gate/up/down, and potentially down-proj separately;
2. scale granularity or scale dtype where practical;
3. layer count and layer selection;
4. acceptance criteria: top1 match, top-k overlap, or prompt text-level tolerance.
