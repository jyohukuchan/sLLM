# Qwen3.5 layers 7-9 hidden 3994 row-dot trace

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Generated artifacts:

- `qwen-layer-module-trace-layers7-9-hidden3994-p4p46-inproj.jsonl`
- `qwen-row-dot-sensitivity-layers7-9-hidden3994-p4p46-inproj.json`

Golden-before row-dot sensitivity:

| layer | projection | original_max_abs | optimal_scale | scaled_max_abs | improvement |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | self_attention_o_proj | 0.105299867 | 1.00541347677 | 0.0496769869 | 0.334789597 |
| 9 | mlp_down_proj | 0.0793600213 | 0.985288347906 | 0.0682068663 | 0.277270086 |
| 9 | attention_out_proj | 0.0781332336 | 1.01484931345 | 0.0823951967 | 0.0185917223 |
| 8 | attention_out_proj | 0.0589695170 | 1.01418149527 | 0.0440059153 | 0.147080859 |
| 8 | mlp_down_proj | 0.0470079087 | 1.02047648266 | 0.0490566167 | 0.262502832 |
| 7 | mlp_down_proj | 0.0444230056 | 1.00310367970 | 0.0454871618 | 0.0102780918 |

Token 11 row-dot errors for hidden 3994:

| layer | projection | package row-dot error vs source |
| ---: | --- | ---: |
| 7 | self_attention_o_proj | -0.00885246908 |
| 7 | mlp_down_proj | 0.0120474609 |
| 8 | attention_out_proj | -0.0195115685 |
| 8 | mlp_down_proj | -0.0220584367 |
| 9 | attention_out_proj | -0.0481112904 |
| 9 | mlp_down_proj | 0.0216266042 |

Actual-prefix hidden 3994/token 11 drift after layer6+10 metadata:

| layer | input_diff | output_diff | delta_diff |
| ---: | ---: | ---: | ---: |
| 7 | -0.139934540 | -0.463647842 | -0.323713303 |
| 8 | -0.463647842 | 0.211425781 | 0.675073624 |
| 9 | 0.211425781 | -0.300535202 | -0.511960983 |
| 10 | -0.300535202 | -0.444337845 | -0.143802643 |
| 11 | -0.444337845 | -0.891334534 | -0.446996689 |

Interpretation:

- Layer `7` has the most scale-like final-row signal, but token `11` itself has only a small row-dot error.
- Layer `8` and layer `9` token `11` final-row errors are much smaller than their actual-prefix delta swings.
- The hidden `3994` residual drift is not explained by one final projection row. It is more likely caused by input distribution changes moving through attention/MLP nonlinear paths.
