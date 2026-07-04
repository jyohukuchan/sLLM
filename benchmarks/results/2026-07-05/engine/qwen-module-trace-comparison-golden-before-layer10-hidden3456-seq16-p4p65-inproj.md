| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | -0.886669 | 22.75 | 21.8633 | 22.75 | -0.886669 | -0.169057 | -0.0558117 | -0.613533 | -0.1482 | ok | -4.55987e-05 | 1673:0.142504 | ok | 0.000325346 | 5301:0.233553 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | attention_a_projection | ok | -0.000639852 | -0.00332553 | -0.084044 | 18:-0.10851 | 22:0.0843817 |
| 10 | 0 | 3456 | attention_b_projection | ok | 0.00416355 | -0.00352774 | 0.0564318 | 24:-0.12673 | 24:-0.12673 |
| 10 | 0 | 3456 | attention_beta | ok | 0.00146771 | 0.00119629 | 9.31025e-05 | 22:-0.00083971 | 22:-0.00083971 |
| 10 | 0 | 3456 | attention_conv | ok | -1.33925e-05 | -0.000182285 | -0.000777245 | 3221:-0.0103699 | 6968:-0.00773668 |
| 10 | 0 | 3456 | attention_conv_pre_silu | ok | -3.69729e-05 | -0.000741481 | -0.0243096 | 428:0.0529044 | 5769:0.0097332 |
| 10 | 0 | 3456 | attention_gate | ok | -0.00190998 | -0.00360375 | -0.00890255 | 19:0.041001 | 13:0.029087 |
| 10 | 0 | 3456 | attention_gate_silu | ok | -0.00164096 | -0.0034969 | 0.0370865 | 3531:-0.136633 | 3133:-0.0614181 |
| 10 | 0 | 3456 | attention_input_normed | ok | 3.71349e-05 | 0.000478762 | 0.0561371 | 3456:0.0561371 | 2872:0.000500232 |
| 10 | 0 | 3456 | attention_pre_gate_normed | ok | -0.000242528 | -0.00163332 | 0.0114231 | 2079:-0.146273 | 1768:-0.0337968 |
| 10 | 0 | 3456 | attention_projection_input | ok | -4.55987e-05 | -0.00219033 | -0.142504 | 1673:0.142504 | 1673:0.142504 |
| 10 | 0 | 3456 | attention_qkv_projection | ok | -0.000844861 | -0.00697529 | -0.0618134 | 4985:-0.181145 | 5769:-0.144447 |
| 10 | 0 | 3456 | attention_recurrent | ok | -1.18925e-06 | -8.63769e-06 | -0.000421047 | 3595:-0.000421047 | 1768:-0.000223951 |
| 10 | 0 | 3456 | attention_recurrent_k | ok | 5.91966e-08 | 4.47993e-08 | 0.00360847 | 1173:-0.0135499 | 1597:-0.00491393 |
| 10 | 0 | 3456 | attention_recurrent_q | ok | 2.27495e-05 | 2.97926e-10 | 0.000466201 | 1159:0.000484198 | 863:0.000376393 |
| 10 | 0 | 3456 | attention_recurrent_v | ok | -9.82292e-06 | -0.000152998 | -0.000777245 | 2079:-0.00887948 | 2872:-0.00773668 |
| 10 | 0 | 3456 | attention_z_projection | ok | -0.00481679 | -0.00681903 | 0.0370569 | 3327:0.184861 | 1768:0.0971797 |
| 10 | 0 | 3456 | mlp_activation | ok | 0.000325346 | -0.00220618 | -0.233553 | 5301:0.233553 | 5301:0.233553 |

## Skipped

| layer | reason | package_hidden | fullref_hidden | token |
|---:|---|---:|---:|---:|
| 0 | missing_full_reference | - | - | - |
| 1 | missing_full_reference | - | - | - |
| 2 | missing_full_reference | - | - | - |
| 3 | missing_full_reference | - | - | - |
| 4 | missing_full_reference | - | - | - |
| 5 | missing_full_reference | - | - | - |
| 6 | missing_full_reference | - | - | - |
| 7 | missing_full_reference | - | - | - |
| 8 | missing_full_reference | - | - | - |
| 9 | missing_full_reference | - | - | - |
| 11 | missing_full_reference | - | - | - |
