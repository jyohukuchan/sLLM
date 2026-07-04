# Qwen Row-Scale Multi-Fixture Summary

## Scope

This summary consolidates the layer6 hidden3994 row-scale and related probes across the available `seq16`, `0..12` Qwen3.5 fixtures.

Common settings:

- Package family: p4p46 in-projection package with existing layer6/layer10 hidden3456 manifest row-scales.
- Run mode: `actual_prefix`
- Layers: `0..12`
- Rotary: `rotary_dim=64`, `rope_base=10000000`, `position_offset=0`
- Backend: CPU

## Full-Prefix Results

| fixture | baseline | layer6 h3994 row-scale | layer8 qkv V845 cell | combined layer6+qkv | extra probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| token ids `1..16` | 0.645338058 | 0.637172699 | 0.627647400 | 0.610977173 | layer10 down row-scale: 0.616283417 |
| token ids `101..116` | 1.080525398 | 1.043153763 | 1.080525398 | 1.043153763 | - |
| token ids `201..216` | 1.140727997 | 1.145284653 | - | - | layer6 + layer11 o row-scale: 1.206287384 |

## Layer6 Row-Scale Effect

| fixture | layer6 before | layer6 after | layer7 before | layer7 after | final effect |
| --- | ---: | ---: | ---: | ---: | --- |
| token ids `1..16` | 0.480636597 | 0.465695381 | 0.627647400 | 0.428003311 | improves final `0.645338058 -> 0.637172699` |
| token ids `101..116` | 0.714679718 | 0.652941704 | 1.080525398 | 1.043153763 | improves final `1.080525398 -> 1.043153763` |
| token ids `201..216` | 0.537414551 | 0.476898193 | 0.966460228 | 0.497438431 | worsens final `1.140727997 -> 1.145284653` |

## Row-Dot Fit Evidence

| fixture | row | optimal_scale | RMSE before | RMSE after |
| --- | --- | ---: | ---: | ---: |
| token ids `1..16` | layer6 `mlp.down_proj[3994]` | 1.026471714 | 0.117735388 | 0.063680278 |
| token ids `101..116` | layer6 `mlp.down_proj[3994]` | 1.023383096 | 0.131756300 | 0.061972585 |
| token ids `201..216` | layer11 `self_attn.o_proj[3994]` under layer6 row-scale | 0.984954853 | 0.059855519 | 0.024166208 |

## Rejected Probes

| probe | fixture | result | reason |
| --- | --- | ---: | --- |
| layer10 `mlp.gate_proj[9256,3994]` source-restore cell | token ids `1..16` | 0.625913620 | worsened current best `0.610977173` |
| layer10 `mlp.down_proj[3994]` row-scale | token ids `1..16` | 0.616283417 | worsened current best despite row-dot RMSE improvement |
| layer11 `self_attn.o_proj[3994]` row-scale | token ids `201..216` | 1.206287384 | worsened layer6 row-scale condition `1.145284653` |

## Interpretation

- Yes, this remains worth debugging.
- The issue is concentrated and traceable, not random backend noise.
- Layer6 `mlp.down_proj[3994]` row-scale is a real local compensation candidate:
  - it reduces layer6/layer7 hidden3994 drift on all checked fixtures
  - its fitted scale is stable across token ids `1..16` and `101..116`
  - it can be represented as package manifest metadata
- It is not safe as an unconditional promotion policy yet:
  - token ids `201..216` improve at layer6/layer7 but worsen the final layer11 max slightly
  - local row-dot RMSE improvements can worsen full-prefix output, as shown by layer10 and layer11 rejected probes
- The next durable fix should use a multi-fixture objective and treat row-scale as a candidate policy with acceptance gates, not as a one-off correction promoted from a single fixture.
