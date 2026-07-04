# Qwen3.5 layer 11 self-attention row-dot trace

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Generated traces:

- `qwen-layer-module-trace-layer11-hidden3377-full-attn-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-layer11-hidden3994-full-attn-p4p46-inproj.jsonl`
- `qwen-row-dot-sensitivity-layer11-hidden3377-3994-full-attn-p4p46-inproj.json`
- `qwen-self-attention-propagation-layer11-hidden3377-3994-3456-p4p46-inproj.json`

Layer replay check:

- `fixture_match.max_abs_diff = 0.03125`
- `fixture_match.mean_abs_diff = 0.000389998604`
- `fixture_match.mse = 0.000000643706810`

Final-row package row-dot sensitivity:

| hidden | projection | original_rmse | original_max_abs | optimal_scale | scaled_max_abs | improvement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3994 | self_attention_o_proj | 0.0749885795 | 0.106661194 | 0.981636672441 | 0.107175835 | 0.448467721 |
| 3994 | mlp_down_proj | 0.023801153 | 0.0381102959 | 1.00060853967 | 0.0384955943 | 0.00140691117 |
| 3377 | mlp_down_proj | 0.00762573429 | 0.0151258546 | 0.994034435104 | 0.0151908693 | 0.0310453531 |
| 3377 | self_attention_o_proj | 0.00428887128 | 0.0113390533 | 0.971243529658 | 0.00737102677 | 0.178608395 |

Projection row-dot hotspots:

| hidden | projection | worst package row-dot error |
| ---: | --- | ---: |
| 3377 | self_attention_q_projection | 0.453516544 |
| 3377 | self_attention_k_projection | -0.144860456 |
| 3377 | self_attention_v_projection | -0.159570191 |
| 3994 | self_attention_q_projection | 0.295013563 |
| 3994 | self_attention_k_projection | 0.507676098 |
| 3994 | self_attention_v_projection | -0.987190539 |

Package q/k/v propagation through attention:

| stage | mse | mean_abs | max_abs |
| --- | ---: | ---: | ---: |
| package_q_projection_vs_source | 0.00508335839 | 0.0549604281 | 0.823976994 |
| package_k_projection_vs_source | 0.00498192002 | 0.0538277185 | 0.479312897 |
| package_v_projection_vs_source | 0.00267562417 | 0.040563034 | 0.952980042 |
| package_o_input_vs_source | 0.000112062289 | 0.00615733091 | 0.187569141 |

Hidden-row propagation summary:

| hidden | worst input token | input error via source o row | worst total token | total error via package o row |
| ---: | ---: | ---: | ---: | ---: |
| 3377 | 10 | -0.0221332256 | 1 | 0.0242107697 |
| 3994 | 14 | -0.0975656509 | 13 | 0.11529398 |
| 3456 | 1 | 0.00994926319 | 4 | -0.0106792711 |

Actual-prefix layer 11 comparison:

| package/run | layer11 input max abs | input location | layer11 output max abs | output location |
| --- | ---: | --- | ---: | --- |
| no metadata, CPU | 1.744266510 | token 0, hidden 3456 | 1.686901093 | token 0, hidden 3456 |
| manifest row-scale, CPU | 0.967845917 | token 0, hidden 3456 | 0.911422729 | token 0, hidden 3456 |
| manifest row-scale, R9700 | 0.967796326 | token 0, hidden 3456 | 0.911373138 | token 0, hidden 3456 |

Interpretation:

- Layer `11` full-attention replay is close enough to the golden fixture for module-level debugging.
- The final `self_attention_o_proj` and `mlp_down_proj` row errors are much smaller than the layer `10` row-scale target.
- One-row scaling is not the next best lever for layer `11`.
- The larger raw errors are in q/k/v projection checks, especially `self_attention_v_projection` for hidden `3994`, but the causal attention mix reduces the full `o_proj` input max abs difference to about `0.188`.
- Hidden `3994` still receives a measurable `o_proj` input contribution (`0.098` using the source row, `0.115` including the package row), so the next target is attention-input propagation rather than blind final-row scale overrides.
- The largest actual-prefix layer `11` error is at hidden `3456`, where the layer input already carries almost the same max difference as the layer output. The layer-local propagation diagnostic only shows about `0.011` local hidden `3456` contribution, so this max coordinate is mainly inherited residual drift, not a new layer `11` row-scale target.
