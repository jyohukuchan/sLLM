| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | -0.875896 | 22.75 | 21.8741 | 22.75 | -0.875896 | -0.169057 | -0.0211814 | -0.613533 | -0.172058 | ok | 8.14793e-05 | 3133:-0.0628624 | ok | 0.000256506 | 5301:0.287889 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | attention_a_projection | ok | -0.000639852 | -0.00332553 | -0.084044 | 18:-0.10851 | 22:0.0843817 |
| 10 | 0 | 3456 | attention_b_projection | ok | 0.00416355 | -0.00352774 | 0.0564318 | 24:-0.12673 | 24:-0.12673 |
| 10 | 0 | 3456 | attention_beta | ok | 0.00146771 | 0.00119629 | 9.31025e-05 | 22:-0.00083971 | 22:-0.00083971 |
| 10 | 0 | 3456 | attention_conv | ok | -2.13754e-05 | -0.000206748 | -0.0182368 | 3221:-0.0780796 | 6968:-0.0138521 |
| 10 | 0 | 3456 | attention_conv_pre_silu | ok | -5.97646e-05 | -0.000591608 | -0.10644 | 2125:0.10644 | 6968:-0.0143623 |
| 10 | 0 | 3456 | attention_gate | ok | -0.00190998 | -0.00360375 | -0.00890255 | 19:0.041001 | 13:0.029087 |
| 10 | 0 | 3456 | attention_gate_silu | ok | -0.000904397 | -0.00272032 | -0.105514 | 2899:-0.105514 | 1759:0.0774374 |
| 10 | 0 | 3456 | attention_input_normed | ok | 3.71349e-05 | 0.000478762 | 0.0561371 | 3456:0.0561371 | 2872:0.000500232 |
| 10 | 0 | 3456 | attention_pre_gate_normed | ok | -0.000124594 | 0.000388233 | -0.126117 | 2079:0.192911 | 2872:-0.126117 |
| 10 | 0 | 3456 | attention_projection_input | ok | 8.14793e-05 | -0.00062872 | -0.0305443 | 3133:-0.0628624 | 3133:-0.0628624 |
| 10 | 0 | 3456 | attention_qkv_projection | ok | -0.0015075 | -0.00758047 | -0.111279 | 2164:-0.19987 | 7229:-0.269696 |
| 10 | 0 | 3456 | attention_recurrent | ok | -4.30174e-07 | -1.42933e-06 | 0.000274995 | 2872:-0.000751255 | 2872:-0.000751255 |
| 10 | 0 | 3456 | attention_recurrent_k | ok | 7.01982e-05 | -3.16631e-09 | 0.00563705 | 1173:-0.0441145 | 858:-0.00113611 |
| 10 | 0 | 3456 | attention_recurrent_q | ok | -9.59007e-06 | -6.31275e-10 | -0.000693675 | 278:0.00338363 | 863:0.00039902 |
| 10 | 0 | 3456 | attention_recurrent_v | ok | -1.78413e-05 | -0.000275269 | -0.0182368 | 3595:-0.0182368 | 2872:-0.0138521 |
| 10 | 0 | 3456 | attention_z_projection | ok | -0.00189163 | -0.00344628 | -0.10542 | 3327:0.116791 | 1759:0.0711145 |
| 10 | 0 | 3456 | mlp_activation | ok | 0.000256506 | -0.00268477 | -0.287889 | 5301:0.287889 | 5301:0.287889 |

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
