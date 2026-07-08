# SQ FP8 `kup6_gate5_down5` Policy v0.1

Date: 2026-07-08

## Summary

This file records the current six-layer FP8 SQ regression subset as a policy artifact.

It is not the final SQ policy.

## FP8 Selection

Scale:

- granularity: `row_block`
- block columns: `32`
- scale dtype: `f32`

Selected FP8 tensors:

| family | layers |
| --- | --- |
| `self_attn.k_proj` | `3,7,11,15,19,23` |
| `mlp.up_proj` | `3,7,11,15,19,23` |
| `mlp.gate_proj` | `3,7,11,15,19` |
| `mlp.down_proj` | `3,7,11,15,19` |

Expected FP8 tensor count: `22`.

Builder include regex:

```text
^model\.language_model\.layers\.((3|7|11|15|19|23)\.(self_attn\.k_proj|mlp\.up_proj)|(3|7|11|15|19)\.mlp\.(gate_proj|down_proj))\.weight$
```

## Fallback Policy

| family | layers | reason |
| --- | --- | --- |
| `self_attn.q_proj` | all tested | strict top1 risk under row-block32 |
| `self_attn.v_proj` | all tested | strict top1 risk not recovered by tested row-block sizes |
| `self_attn.o_proj` | all tested | six-layer cumulative drift boundary |
| `mlp.gate_proj` | `23` | layer 23 six-layer boundary |
| `mlp.down_proj` | `23` | layer 23 six-layer boundary |
| all other language model tensors | all | not selected in current six-layer regression subset |
| visual/MTP/RDNA2/tensor-parallel targets | all | deferred |

## Prompt Bundle Result

| case | strict top1 | top8 common |
| --- | --- | ---: |
| len4 | true | `7 / 8` |
| case_a | true | `2 / 8` |
| case_b | true | `6 / 8` |

The case_a top8 overlap is low, so this policy remains a regression subset rather than a full SQ
policy.

## Next Action

1. Use this policy as the current six-layer strict-top1 regression subset.
2. Keep `kup6_ogatedown5` as a near-miss failure guard.
3. Move T1 real batch runner forward before using throughput rows for SQ performance decisions.
