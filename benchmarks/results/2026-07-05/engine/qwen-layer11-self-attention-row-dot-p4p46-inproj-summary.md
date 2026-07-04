# Qwen3.5 layer 11 self-attention row-dot trace

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Generated traces:

- `qwen-layer-module-trace-layer11-hidden3377-full-attn-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-layer11-hidden3994-full-attn-p4p46-inproj.jsonl`
- `qwen-row-dot-sensitivity-layer11-hidden3377-3994-full-attn-p4p46-inproj.json`

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

Interpretation:

- Layer `11` full-attention replay is close enough to the golden fixture for module-level debugging.
- The final `self_attention_o_proj` and `mlp_down_proj` row errors are much smaller than the layer `10` row-scale target.
- One-row scaling is not the next best lever for layer `11`.
- The larger measurable errors are in q/k/v projection row-dot checks, especially `self_attention_v_projection` for hidden `3994`.
- Next debugging should trace q/k/v projection error through q/k norm, RoPE, attention value mix, gate, and `o_proj` input rather than adding more final-row scale overrides.
