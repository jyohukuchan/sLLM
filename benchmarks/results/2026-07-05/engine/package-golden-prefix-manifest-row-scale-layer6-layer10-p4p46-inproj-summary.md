# Qwen3.5 p4p46 manifest row-scale layer 6 + layer 10

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Source package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Metadata entries:

| tensor | row | scale |
| --- | ---: | ---: |
| `model.language_model.layers.6.linear_attn.out_proj.weight` | 3456 | 1.032273364777375 |
| `model.language_model.layers.6.mlp.down_proj.weight` | 3456 | 1.036585679248007 |
| `model.language_model.layers.10.linear_attn.out_proj.weight` | 3456 | 1.0230717930961908 |
| `model.language_model.layers.10.mlp.down_proj.weight` | 3456 | 1.0416570117172528 |

Run summary:

| run | backend | max_mse | max_mse_layer | max_mean_abs | max_abs | max_abs_layer | max_abs_location | min_cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| golden_before_each_layer | CPU | 0.000740506879 | 11 | 0.020715800 | 0.508314133 | 7 | token 15, hidden 3994 | 0.998585695 |
| actual_prefix | CPU | 0.004097481631 | 11 | 0.050090760 | 0.891334534 | 11 | token 11, hidden 3994 | 0.992993845 |
| golden_before_each_layer | R9700 | 0.000740507114 | 11 | 0.020715803 | 0.508314133 | 7 | token 15, hidden 3994 | 0.998585695 |
| actual_prefix | R9700 | 0.004097484565 | 11 | 0.050090781 | 0.891326904 | 11 | token 11, hidden 3994 | 0.992993839 |

Comparison against layer 10 only manifest metadata:

| run | backend | layer10-only max_abs | layer6+10 max_abs | delta |
| --- | --- | ---: | ---: | ---: |
| golden_before_each_layer | CPU | 0.508314133 | 0.508314133 | 0.000000000 |
| actual_prefix | CPU | 0.967845917 | 0.891334534 | -0.076511383 |
| golden_before_each_layer | R9700 | 0.508314133 | 0.508314133 | 0.000000000 |
| actual_prefix | R9700 | 0.967796326 | 0.891326904 | -0.076469421 |

Layer-local actual-prefix movement:

| layer | layer10-only max_abs | layer10-only location | layer6+10 max_abs | layer6+10 location |
| ---: | ---: | --- | ---: | --- |
| 6 | 0.588710785 | token 0, hidden 3456 | 0.480636597 | token 0, hidden 3994 |
| 8 | 0.735359192 | token 0, hidden 3456 | 0.476449966 | token 8, hidden 3994 |
| 10 | 0.967845917 | token 0, hidden 3456 | 0.461685181 | token 7, hidden 3994 |
| 11 | 0.911422729 | token 0, hidden 3456 | 0.891334534 | token 11, hidden 3994 |

Interpretation:

- Layer `6` row-scale metadata removes the hidden `3456` actual-prefix chain that previously survived into layer `11`.
- The overall actual-prefix max abs improves modestly, from about `0.9678` to about `0.8913`.
- The remaining dominant coordinate moves to hidden `3994`, matching the layer `11` self-attention propagation diagnosis.
- CPU and R9700 agree within expected backend tolerance.
