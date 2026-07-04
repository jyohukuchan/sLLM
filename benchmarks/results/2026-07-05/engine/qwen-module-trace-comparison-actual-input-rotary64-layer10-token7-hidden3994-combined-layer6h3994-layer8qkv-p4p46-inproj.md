| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7 | 3994 | -0.362198 | -0.5 | -0.577299 | -0.715101 | 0.137802 | 0.00249964 | -0.0112711 | 0.0519822 | 0.105654 | ok | 0.000288043 | 3593:-0.083459 | ok | -0.000123365 | 9256:-0.187328 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 7 | 3994 | attention_a_projection | ok | -0.00301113 | 0.000581089 | 0.031301 | 16:-0.0557115 | 19:0.0624243 |
| 10 | 7 | 3994 | attention_b_projection | ok | -0.0138815 | -0.00750836 | -0.0379391 | 16:-0.17312 | 17:0.125919 |
| 10 | 7 | 3994 | attention_beta | ok | 0.00227356 | 0.00201436 | -0.000200033 | 27:0.020041 | 19:-0.0221188 |
| 10 | 7 | 3994 | attention_conv | ok | -0.00013909 | -0.00059221 | 0.0541135 | 93:-0.117476 | 7691:-0.105235 |
| 10 | 7 | 3994 | attention_conv_pre_silu | ok | -0.000319629 | -0.00137776 | -0.198579 | 3301:0.214306 | 7691:-0.111042 |
| 10 | 7 | 3994 | attention_gate | ok | -0.00119994 | -0.00670656 | -0.0381277 | 28:0.0381277 | 28:0.0381277 |
| 10 | 7 | 3994 | attention_gate_silu | ok | -0.000776829 | -0.00135147 | -0.0343637 | 3588:0.0933363 | 1607:-0.0548491 |
| 10 | 7 | 3994 | attention_input_normed | ok | 0.000482517 | 0.00074693 | 0.0393944 | 3994:0.0393944 | 3595:-0.00378269 |
| 10 | 7 | 3994 | attention_pre_gate_normed | ok | 0.000337035 | -0.000590216 | 0.108952 | 1768:0.134764 | 1768:0.134764 |
| 10 | 7 | 3994 | attention_projection_input | ok | 0.000288043 | 0.00108597 | 0.0155708 | 3593:-0.083459 | 3593:-0.083459 |
| 10 | 7 | 3994 | attention_qkv_projection | ok | -0.00214363 | -0.00339921 | -0.30261 | 2722:-0.30261 | 7691:-0.484842 |
| 10 | 7 | 3994 | attention_recurrent | ok | 1.82792e-07 | -8.5272e-06 | -0.00151571 | 3595:-0.00151571 | 3595:-0.00151571 |
| 10 | 7 | 3994 | attention_recurrent_k | ok | -2.58021e-05 | -6.93939e-09 | 0.00137722 | 1712:0.0103129 | 839:-0.0229932 |
| 10 | 7 | 3994 | attention_recurrent_q | ok | -1.69203e-07 | -9.14398e-11 | 0.00046733 | 1042:0.0027744 | 872:-0.000442551 |
| 10 | 7 | 3994 | attention_recurrent_v | ok | -7.99518e-05 | -0.000486205 | 0.0267262 | 3595:-0.105235 | 3595:-0.105235 |
| 10 | 7 | 3994 | attention_z_projection | ok | -0.00266457 | -0.00512959 | -0.288118 | 36:0.288118 | 1607:-0.0506132 |
| 10 | 7 | 3994 | mlp_activation | ok | -0.000123365 | 0.000212574 | 0.0282688 | 9256:-0.187328 | 9256:-0.187328 |
| 10 | 7 | 3994 | mlp_gate_projection | ok | -0.00124133 | -0.00149153 | 0.0100942 | 547:-0.0497742 | 9256:0.109216 |
| 10 | 7 | 3994 | mlp_gate_silu | ok | -0.000325429 | -0.000512476 | 0.00975227 | 6795:0.0534272 | 9256:0.0907042 |
| 10 | 7 | 3994 | mlp_up_projection | ok | -0.000194618 | -0.000449162 | 0.0630083 | 4977:0.0630083 | 4594:0.035336 |

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
